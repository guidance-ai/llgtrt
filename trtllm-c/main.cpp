#include <stdexcept>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdio>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include "tensorrt_llm/runtime/iTensor.h"
#include <NvInferRuntime.h>

#include "tlc.h"

#define TRY try

#define FINISH                                                                                                         \
    catch (std::exception const& e)                                                                                    \
    {                                                                                                                  \
        return strdup(e.what());                                                                                       \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        return strdup("Unknown exception.");                                                                           \
    }                                                                                                                  \
    return nullptr

namespace tle = tensorrt_llm::executor;
namespace fs = std::filesystem;

struct ResponseData
{
    std::string error;
    tle::VecTokens tokens;
    tle::VecLogProbs logprobs;
    tle::Tensor logitsTensor;
};

struct TlcExecutor
{
    tle::Executor executor;
    bool has_logits_post_processor;
    std::vector<TlcResponse> responses;
    // this is the same size as responses, and is used to keep pointers in responses alive
    std::vector<ResponseData> responses_data;
};

void tlc_default_init_params(TlcInitParams* params)
{
    memset(params, 0, sizeof(TlcInitParams));
    auto e = &params->engine_params;
    e->iter_stats_max_iterations = 1000;
    e->max_batch_size = 128;
    e->max_num_tokens = 8192;
    e->max_queue_delay_microseconds = 3 * 60 * 1000 * 1000;
    e->gpu_weights_percent = 1.0;
    e->kv_cache_free_gpu_mem_fraction = 0.9;
    e->kv_cache_onboard_blocks = true;
    e->cross_kv_cache_fraction = NAN;
    e->secondary_offload_min_priority = INT32_MIN;
    e->event_buffer_max_size = 0;
}

void _tlc_set_logits_post_processor(TlcInitParams const* params, tle::ExecutorConfig* config);

template <typename T>
static std::optional<T> positive_opt(T value)
{
    return value > 0 ? std::optional<T>(value) : std::nullopt;
}

static std::optional<float> nan_opt(float value)
{
    return std::isnan(value) ? std::nullopt : std::optional<float>(value);
}

TlcStatus tlc_init(TlcInitParams const* params, TlcExecutor** res)
{
    TRY
    {
        *res = nullptr;

        initTrtLlmPlugins();

        // all reqs in batch have to have the same beam width, so we just ignore the feature
        int beam_width = 1;

        auto executorConfig = tle::ExecutorConfig(beam_width);
        _tlc_set_logits_post_processor(params, &executorConfig);

        auto ep = &params->engine_params;

        std::optional<std::vector<int32_t>> maxAttentionWindowVec = std::nullopt;
        if (ep->max_attention_window_size > 0)
        {
            maxAttentionWindowVec = std::vector<int32_t>{ep->max_attention_window_size};
        }

        auto kvConfig = tle::KvCacheConfig(ep->enable_kv_cache_reuse, positive_opt(ep->max_tokens_in_paged_kv_cache),
            maxAttentionWindowVec, positive_opt(ep->sink_token_length),
            ep->kv_cache_free_gpu_mem_fraction > 0 && ep->max_tokens_in_paged_kv_cache <= 0
                ? std::optional<float>(ep->kv_cache_free_gpu_mem_fraction)
                : std::nullopt,
            ep->kv_cache_host_memory_bytes, ep->kv_cache_onboard_blocks, nan_opt(ep->cross_kv_cache_fraction),
            ep->secondary_offload_min_priority == INT32_MIN
                ? std::nullopt
                : std::optional<int32_t>(ep->secondary_offload_min_priority),
            ep->event_buffer_max_size);

        tle::DynamicBatchConfig dynamicBatchConfig(ep->enable_batch_size_tuning, ep->enable_max_num_tokens_tuning);

        auto policy = ep->guaranteed_no_evict ? tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT
                                              : tle::CapacitySchedulerPolicy::kMAX_UTILIZATION;
        auto chunking = tle::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED; // default?
        auto schedulerConfig = tle::SchedulerConfig(policy, chunking, dynamicBatchConfig);

        executorConfig.setKvCacheConfig(kvConfig);
        executorConfig.setSchedulerConfig(schedulerConfig);

        executorConfig.setEnableChunkedContext(ep->enable_chunked_context);
        executorConfig.setMaxBatchSize(ep->max_batch_size);
        executorConfig.setMaxNumTokens(ep->max_num_tokens);
        executorConfig.setMaxQueueSize(ep->max_queue_size);
        executorConfig.setMaxSeqIdleMicroseconds(ep->max_queue_delay_microseconds);
        executorConfig.setNormalizeLogProbs(false);

        auto executor
            = new TlcExecutor{tle::Executor(params->engine_path, tle::ModelType::kDECODER_ONLY, executorConfig)};

        if (params->logits_post_processor)
            executor->has_logits_post_processor = true;

        *res = executor;
    }
    FINISH;
}

bool tlc_can_enqueue_request(TlcExecutor* ctx)
{
    return ctx->executor.canEnqueueRequests();
}

void tlc_shutdown(TlcExecutor* ctx)
{
    ctx->executor.shutdown();
}

tle::Shape _tlc_to_tle_shape(TlcShape tlc_shape)
{
    return tle::Shape(tlc_shape.dims, tlc_shape.num_dims);
}

TlcShape _tle_to_tlc_shape(const tle::Shape& tle_shape)
{
    TlcShape tlc_shape;
    tlc_shape.num_dims = std::min(tle_shape.size(), static_cast<size_t>(TLC_MAX_SHAPE)); // Ensure within max shape

    std::memcpy(tlc_shape.dims, tle_shape.begin(), tlc_shape.num_dims * sizeof(int64_t));

    return tlc_shape;
}

static tle::DataType to_tle_datatype(TlcDataType t)
{
    switch (t)
    {
    case TLC_DT_BOOL: return tle::DataType::kBOOL;
    case TLC_DT_U8: return tle::DataType::kUINT8;
    case TLC_DT_I8: return tle::DataType::kINT8;
    case TLC_DT_I32: return tle::DataType::kINT32;
    case TLC_DT_I64: return tle::DataType::kINT64;
    case TLC_DT_BF16: return tle::DataType::kBF16;
    case TLC_DT_F8: return tle::DataType::kFP8;
    case TLC_DT_F16: return tle::DataType::kFP16;
    case TLC_DT_F32: return tle::DataType::kFP32;
    default: throw std::runtime_error("Unsupported data type");
    }
}

static TlcDataType to_tlc_datatype(tle::DataType t)
{
    switch (t)
    {
    case tle::DataType::kBOOL: return TLC_DT_BOOL;
    case tle::DataType::kUINT8: return TLC_DT_U8;
    case tle::DataType::kINT8: return TLC_DT_I8;
    case tle::DataType::kINT32: return TLC_DT_I32;
    case tle::DataType::kINT64: return TLC_DT_I64;
    case tle::DataType::kBF16: return TLC_DT_BF16;
    case tle::DataType::kFP8: return TLC_DT_F8;
    case tle::DataType::kFP16: return TLC_DT_F16;
    case tle::DataType::kFP32: return TLC_DT_F32;
    default: throw std::runtime_error("Unsupported data type");
    }
}

static nvinfer1::DataType to_nvinfer_datatype(TlcDataType t)
{
    static_assert((int) TLC_DT_F32 == (int) nvinfer1::DataType::kFLOAT);
    static_assert((int) TLC_DT_F16 == (int) nvinfer1::DataType::kHALF);
    static_assert((int) TLC_DT_I8 == (int) nvinfer1::DataType::kINT8);
    static_assert((int) TLC_DT_I32 == (int) nvinfer1::DataType::kINT32);
    static_assert((int) TLC_DT_BOOL == (int) nvinfer1::DataType::kBOOL);
    static_assert((int) TLC_DT_U8 == (int) nvinfer1::DataType::kUINT8);
    static_assert((int) TLC_DT_F8 == (int) nvinfer1::DataType::kFP8);
    static_assert((int) TLC_DT_BF16 == (int) nvinfer1::DataType::kBF16);
    static_assert((int) TLC_DT_I64 == (int) nvinfer1::DataType::kINT64);
    static_assert((int) TLC_DT_I4 == (int) nvinfer1::DataType::kINT4);

    if ((unsigned) t > (unsigned) nvinfer1::DataType::kINT4)
        throw std::runtime_error("Unsupported data type");

    return (nvinfer1::DataType) t;
}

tle::Tensor _tlc_to_tle_tensor(TlcTensor tlc_tensor)
{
    // The copyToCpu call at the end is required to make this Tensor manage its own memory
    return tle::Tensor::of(to_tle_datatype(tlc_tensor.data_type), const_cast<void*>(tlc_tensor.data_ptr),
        _tlc_to_tle_shape(tlc_tensor.shape))
        .copyToCpu();
}

tle::Tensor _tlc_to_tle_tensor_no_copy(TlcTensor tlc_tensor)
{
    TLLM_CHECK_WITH_INFO(tlc_tensor.shape.num_dims <= nvinfer1::Dims::MAX_DIMS, "Number of dimensions is too large");
    nvinfer1::Dims shape{};
    shape.nbDims = tlc_tensor.shape.num_dims;
    std::copy(tlc_tensor.shape.dims, tlc_tensor.shape.dims + tlc_tensor.shape.num_dims, shape.d);
    auto itensor = tensorrt_llm::runtime::ITensor::wrap(
        (void*) tlc_tensor.data_ptr, to_nvinfer_datatype(tlc_tensor.data_type), shape);
    return tle::detail::ofITensor(std::move(itensor));
}

static void check_dtype(TlcTensor t, TlcDataType expected, char const* context)
{
    if (t.data_type != expected)
    {
        throw std::runtime_error(std::string("Expected ") + context + " to have data type " + std::to_string(expected)
            + ", got " + std::to_string(t.data_type));
    }
}

static bool tlc_tensor_is_none(TlcTensor const& tlc_tensor)
{
    return tlc_tensor.data_ptr == nullptr;
}

static size_t tlc_shape_volume(TlcShape const& shape)
{
    size_t volume = 1;
    for (size_t i = 0; i < shape.num_dims; ++i)
        volume *= shape.dims[i];
    return volume;
}

TlcStatus tlc_enqueue_request(TlcExecutor* ctx, TlcRequest const* request, TlcReqId* res)
{
    TRY
    {
        *res = 0;
        tle::OutputConfig outputConfig;
        outputConfig.excludeInputFromOutput = true;
        outputConfig.returnLogProbs = request->params.logprobs;

        tle::SamplingConfig samplingConfig;

        // ** code from trtllm, need to map DraftParams to it below
        // think target data types are tle::VecTokens tokens; tle::VecLogProbs logprobs;
        // TODO how spec decoding called?
        // tle::ExternalDraftTokensConfig draftTokensConfig(
        //  std::move(draftTokens), logitsTensor, std::nullopt /* acceptance threshold */, runtimeOpts.fastLogits);
        // request.setExternalDraftTokensConfig(draftTokensConfig);

        auto const& p = request->params;
        auto const& pp = request->prompt_params;

        if (std::isfinite(p.temperature))
            samplingConfig.setTemperature(p.temperature);
        if (std::isfinite(p.top_p))
            samplingConfig.setTopP(p.top_p);
        if (std::isfinite(p.frequency_penalty))
            samplingConfig.setFrequencyPenalty(p.frequency_penalty);
        if (std::isfinite(p.presence_penalty))
            samplingConfig.setPresencePenalty(p.presence_penalty);
        if (p.seed != UINT64_MAX)
            samplingConfig.setSeed(p.seed);
        if (p.top_k != 0)
            samplingConfig.setTopK(p.top_k);
        if (p.min_tokens != 0)
            samplingConfig.setMinTokens(p.min_tokens);
        samplingConfig.setNumReturnSequences(p.num_return_sequences);

        tle::VecTokens tokens(request->tokens, request->tokens + request->num_tokens);
        tle::Request req(std::move(tokens), p.max_new_tokens, p.streaming, samplingConfig, outputConfig);

        req.setClientId(request->client_req_id);
        req.setPriority(p.priority);

        if (p.eos_token_id != UINT32_MAX)
            req.setEndId(p.eos_token_id);

        if (ctx->has_logits_post_processor && p.use_logits_post_processor)
            req.setLogitsPostProcessorName(tle::Request::kBatchedPostProcessorName);

        if (!tlc_tensor_is_none(pp.prompt_table))
        {
            std::optional<std::vector<tle::IdType>> inputTokenExtraIds = std::nullopt;
            if (!tlc_tensor_is_none(pp.input_token_extra_ids))
            {
                check_dtype(pp.input_token_extra_ids, TLC_DT_I64, "input_token_extra_ids");
                auto ptr = (tle::IdType*) pp.input_token_extra_ids.data_ptr;
                auto len = tlc_shape_volume(pp.input_token_extra_ids.shape);
                inputTokenExtraIds = std::vector(ptr, ptr + len);
            }
            auto ptune = tle::PromptTuningConfig(_tlc_to_tle_tensor_no_copy(pp.prompt_table), inputTokenExtraIds);
            req.setPromptTuningConfig(ptune);
        }

        if (!tlc_tensor_is_none(pp.mrope_rotary_sin_cos))
        {
            auto mrope
                = tle::MropeConfig(_tlc_to_tle_tensor_no_copy(pp.mrope_rotary_sin_cos), pp.mrope_position_deltas);
            req.setMropeConfig(mrope);
        }

        if (!tlc_tensor_is_none(pp.skip_cross_attn_blocks))
            req.setSkipCrossAttnBlocks(_tlc_to_tle_tensor_no_copy(pp.skip_cross_attn_blocks));

        if (pp.encoder_output_length > 0)
            req.setEncoderOutputLength(pp.encoder_output_length);

        if (!tlc_tensor_is_none(pp.encoder_input_features))
            req.setEncoderInputFeatures(_tlc_to_tle_tensor_no_copy(pp.encoder_input_features));

        if (!tlc_tensor_is_none(pp.cross_attention_masks))
            req.setCrossAttentionMask(_tlc_to_tle_tensor_no_copy(pp.cross_attention_masks));

        if (!tlc_tensor_is_none(pp.input_position_ids))
        {
            check_dtype(pp.input_position_ids, TLC_DT_I32, "input_position_ids");
            auto ptr = (std::int32_t*) pp.input_position_ids.data_ptr;
            auto len = tlc_shape_volume(pp.input_position_ids.shape);
            req.setPositionIds(std::vector(ptr, ptr + len));
        }

        if (request->lora_params.lora_id)
        {
            auto const& lp = request->lora_params;
            std::optional<tle::Tensor> weights;
            if (lp.weights.data_ptr)
            {
                weights = _tlc_to_tle_tensor(lp.weights);
            }
            std::optional<tle::Tensor> config;
            if (lp.config.data_ptr)
            {
                config = _tlc_to_tle_tensor(lp.config);
            }
            tle::LoraConfig loraConfig(lp.lora_id, weights, config);
            req.setLoraConfig(loraConfig);
        }

        // If we have draft params build draft config
        if (request->draft_params.draft_tokens && request->draft_params.logits_tensor.data_ptr)
        {
            auto const& dp = request->draft_params;
            tle::Tensor logitsTensor;
            logitsTensor = _tlc_to_tle_tensor(dp.logits_tensor);
            assert(dp.num_tokens > 0);
            tle::VecTokens draftTokens(dp.draft_tokens, dp.draft_tokens + dp.num_tokens);
            tle::ExternalDraftTokensConfig draftTokensConfig(
                std::move(draftTokens), logitsTensor, std::nullopt, std::nullopt);
            req.setExternalDraftTokensConfig(draftTokensConfig);
        }

        std::vector<tle::Request> requests;
        requests.emplace_back(std::move(req));
        auto ids = ctx->executor.enqueueRequests(std::move(requests));
        *res = ids[0];
    }
    FINISH;
}

TlcStatus tlc_cancel_request(TlcExecutor* ctx, TlcReqId req_id)
{
    TRY
    {
        ctx->executor.cancelRequest(req_id);
    }
    FINISH;
}

TlcStatus tlc_await_responses(
    TlcExecutor* ctx, uint32_t timeout_ms, TlcResponse const** responses, uint32_t* num_responses)
{
    TRY
    {
        ctx->responses.clear();
        ctx->responses_data.clear();

        std::chrono::milliseconds timeout{timeout_ms};
        auto responsesData = ctx->executor.awaitResponses(timeout);

        ctx->responses.reserve(responsesData.size());
        ctx->responses_data.reserve(responsesData.size());

        for (auto const& response : responsesData)
        {
            TlcResponse c_resp = {};
            ResponseData resp_data = {};

            c_resp.req_id = response.getRequestId();
            if (response.hasError())
            {
                std::string err = "ReqId " + std::to_string(response.getRequestId())
                    + " has already been processed and was terminated.";
                if (response.getErrorMsg() == err)
                {
                    // ignore this error; copied from
                    // https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/cpp/executor/executorExampleAdvanced.cpp#L263
                    continue;
                }
                else
                {
                    resp_data.error = response.getErrorMsg();
                }
            }
            else
            {
                auto result = response.getResult();
                c_resp.sequence_idx = result.sequenceIndex;
                c_resp.is_seq_final = result.isSequenceFinal;
                c_resp.is_req_final = result.isFinal;
                c_resp.finish_reason = FINISH_REASON_NOT_FINISHED;
                if (result.finishReasons.size() > 0)
                {
                    switch (result.finishReasons[0])
                    {
                    case tle::FinishReason::kEND_ID: c_resp.finish_reason = FINISH_REASON_END_ID; break;
                    case tle::FinishReason::kLENGTH: c_resp.finish_reason = FINISH_REASON_LENGTH; break;
                    // shouldn't happen - we don't support stop words at this level
                    case tle::FinishReason::kSTOP_WORDS: c_resp.finish_reason = FINISH_REASON_STOP_WORDS; break;
                    default: break;
                    }
                }
                assert(result.outputTokenIds.size() == 1);
                resp_data.tokens = result.outputTokenIds.at(0);

                // Grab generationLogits, TODO=need to see if nonstreaming/streaming matters here
                auto generationLogits = result.generationLogits.value();
                auto logitsShape = generationLogits.getShape();
                assert(logitsShape[0] == 1);
                resp_data.logitsTensor = tle::Tensor::cpu(generationLogits.getDataType(), {logitsShape[1], logitsShape[2]});
                std::memcpy(logitsTensor.getData(), generationLogits.getData(), generationLogits.getSizeInBytes());

                if (result.logProbs.has_value())
                {
                    assert(result.logProbs->size() == 1);
                    auto const& logprobs_ref = result.logProbs->at(0);
                    // take last |tokens| logprobs
                    if (logprobs_ref.size() > resp_data.tokens.size())
                    {
                        resp_data.logprobs.assign(logprobs_ref.end() - resp_data.tokens.size(), logprobs_ref.end());
                    }
                    else
                    {
                        resp_data.logprobs = logprobs_ref;
                    }
                }
            }

            ctx->responses_data.emplace_back(std::move(resp_data));

            auto const& data = ctx->responses_data.back();

            if (data.error.size() > 0)
            {
                c_resp.error = data.error.c_str();
            }
            else
            {
                c_resp.num_tokens = data.tokens.size();
                c_resp.tokens = data.tokens.data();
                c_resp.num_logprobs = data.logprobs.size();
                if (c_resp.num_logprobs > 0)
                    c_resp.logprobs = data.logprobs.data();
                c_resp.logits_tensor.data_type = to_tlc_datatype(data.logitsTensor.getDataType());
                c_resp.logits_tensor.data_ptr = resp_data.logitsTensor.getData();
                c_resp.logits_tensor.shape = _tle_to_tlc_shape(resp_data.logitsTensor.getShape());
            }

            ctx->responses.emplace_back(c_resp);
        }

        *responses = ctx->responses.data();
        *num_responses = ctx->responses.size();
    }
    FINISH;
}