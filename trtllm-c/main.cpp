#include <stdexcept>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdio>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

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
}

void _tlc_set_logits_post_processor(TlcInitParams const* params, tle::ExecutorConfig* config);

template <typename T>
static std::optional<T> positive_opt(T value)
{
    return value > 0 ? std::optional<T>(value) : std::nullopt;
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
            ep->kv_cache_host_memory_bytes, ep->kv_cache_onboard_blocks);

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
    return tle::Shape(tlc_shape.dims_ptr, tlc_shape.num_dims);
}

tle::Tensor _tlc_to_tle_tensor(TlcTensor tlc_tensor)
{
    // The copyToCpu call at the end is required to make this Tensor manage its own memory
    return tle::Tensor::of(static_cast<tle::DataType>(tlc_tensor.data_type), const_cast<void*>(tlc_tensor.data_ptr), _tlc_to_tle_shape(tlc_tensor.shape)).copyToCpu();
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

        auto const& p = request->params;

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

        if (request->lora_params.lora_id) {
            auto const& lp = request->lora_params;
            std::optional<tle::Tensor> weights;
            if (lp.weights.data_ptr) {
                weights = _tlc_to_tle_tensor(lp.weights);
            }
            std::optional<tle::Tensor> config;
            if (lp.config.data_ptr) {
                config = _tlc_to_tle_tensor(lp.config);
            }
            tle::LoraConfig loraConfig(lp.lora_id, weights, config);
            req.setLoraConfig(loraConfig);
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
            }

            ctx->responses.emplace_back(c_resp);
        }

        *responses = ctx->responses.data();
        *num_responses = ctx->responses.size();
    }
    FINISH;
}