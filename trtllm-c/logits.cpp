#include <stdexcept>
#include <string>
#include <cmath>
#include <cassert>

#include "mask_logits.h"

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include "tlc.h"

namespace tle = tensorrt_llm::executor;

static uint64_t lastSyncTime = 0;
static void* masksData;
static void* cudaMasksData;
static int32_t mask_stride;
static int32_t max_batch_size;
static int64_t masks_size;
static TlcLogitsPostProcessor logits_post_processor;

void* tlc_alloc_logit_data(int32_t mask_stride_, int32_t max_batch_size_)
{
    assert(masksData == nullptr);

    mask_stride = mask_stride_;
    max_batch_size = max_batch_size_;

    assert(mask_stride > 0);
    assert(max_batch_size > 0);
    assert(mask_stride % 4 == 0);

    size_t hd_size = max_batch_size * sizeof(int64_t) * 4;
    size_t sz2 = hd_size + max_batch_size * mask_stride;
    masks_size = sz2;
    if (cudaHostAlloc(&masksData, sz2, cudaHostAllocDefault))
        return NULL;
    if (cudaMalloc(&cudaMasksData, sz2))
        return NULL;
    return (uint8_t*) masksData + hd_size;
}

float* tlc_mask_fraction_ptr()
{
    return (float*) ((uint8_t*) masksData + max_batch_size * sizeof(int64_t) * 3);
}

#define MAX_BATCH_SIZE 128

static uint64_t monotimer()
{
    struct timespec ts;
    static struct timespec ts0;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    if (ts0.tv_sec == 0)
    {
        ts0.tv_sec = ts.tv_sec;
    }
    return (ts.tv_sec - ts0.tv_sec) * 1000000000 + ts.tv_nsec;
}

static std::string shapeToString(tle::Shape const& shape)
{
    std::string shapeStr = "[";
    for (size_t i = 0; i < shape.size(); i++)
    {
        shapeStr += std::to_string(shape[i]);
        if (i < shape.size() - 1)
        {
            shapeStr += ", ";
        }
    }
    shapeStr += "]";
    return shapeStr;
}

#define CHECK(x)                                                                                                       \
    if (!(x))                                                                                                          \
    {                                                                                                                  \
        TLLM_LOG_ERROR("CHECK failed: %s", #x);                                                                        \
        abort();                                                                                                       \
    }

static void logitsPostProcessorFn(std::vector<tle::IdType> const& reqIds, std::vector<tle::Tensor>& logits,
    std::vector<std::reference_wrapper<tle::BeamTokens const>> const& tokens, tle::StreamPtr const& streamPtr,
    std::vector<std::optional<tle::IdType>> const& clientIds)
{
    uint64_t start = monotimer();
    // streamPtr->synchronize();

    assert(masksData != nullptr);
    int64_t batchSize = (int64_t) reqIds.size();
    assert(batchSize <= max_batch_size);

    std::vector<TlcLogitsEntry> entries;
    entries.reserve(batchSize);

    for (int64_t i = 0; i < batchSize; ++i)
    {
        TlcLogitsEntry entry = {};
        entry._req_id = reqIds[i];
        entry._client_req_id = clientIds[i].value_or(0);
        assert(tokens[i].get().size() == 1);
        entry._tokens = tokens[i].get()[0].data();
        entry._num_tokens = tokens[i].get()[0].size();
        entry.out_mask_pointer = nullptr;
        entry.temperature = 1.0f;
        entries.push_back(entry);

        // auto shape = logits[i].getShape();
        // TLLM_LOG_INFO("reqId=%d/%d shape=%s n=%d", reqIds[i], entry.client_req_id, shapeToString(shape).c_str(),
        //     entry.num_tokens);
    }

    uint64_t t1 = monotimer();

    logits_post_processor(entries.data(), entries.size());

    uint64_t t2 = monotimer();

    std::string msg = "times: prep=" + std::to_string(t1 - start) + " rust=" + std::to_string(t2 - t1)
        + " step=" + std::to_string(start - lastSyncTime) + " batch=" + std::to_string(reqIds.size());

    CUstream_st* stream = (CUstream_st*) streamPtr->get();

    int64_t* logitPtrs = (int64_t*) masksData;
    int64_t* masksOffsets = logitPtrs + batchSize;
    float* temperatures = (float*) (logitPtrs + 2 * batchSize);

    int64_t temperatures_offset = (uint8_t*) temperatures - (uint8_t*) masksData;
    int64_t mask_fractions_offset = (uint8_t*) tlc_mask_fraction_ptr() - (uint8_t*) masksData;

    int64_t* cudaLogitPtrs = (int64_t*) cudaMasksData;
    int64_t* cudaMasksOffsets = (int64_t*) cudaLogitPtrs + batchSize;

    cudaDataType tp = (cudaDataType) 0;

    size_t dp = 0;
    int64_t max_offset = 0;
    int32_t nVocab = 0;

    for (size_t i = 0; i < reqIds.size(); i++)
    {
        if (entries[i].out_mask_pointer == nullptr)
            continue;

        // beam_size == 1 !
        CHECK(tokens[i].get().size() == 1);
        auto shape = logits[i].getShape();
        CHECK(shape.size() == 3);
        // one of these two is beam size
        CHECK(shape[0] == 1);
        CHECK(shape[1] == 1);
        if (nVocab == 0)
        {
            nVocab = shape[2];

            if ((nVocab + 31) / 32 * 4 > mask_stride)
            {
                TLLM_LOG_ERROR("nVocab=%d; mask_stride allows for only %d", nVocab, mask_stride * 8);
                TLLM_LOG_INFO("try adding { ... tokenizer: { n_vocab_override: %d } ... } to llgtrt.json5", nVocab);
                abort();
            }
        }
        else
        {
            CHECK(nVocab == shape[2]);
        }
        CHECK(logits[i].getMemoryType() == tle::MemoryType::kGPU);

        cudaDataType tp2;
        switch (logits[i].getDataType())
        {
        case tle::DataType::kBF16: tp2 = CUDA_R_16BF; break;
        case tle::DataType::kFP16: tp2 = CUDA_R_16F; break;
        case tle::DataType::kFP32: tp2 = CUDA_R_32F; break;
        default: CHECK(false); break;
        }
        if (!tp)
            tp = tp2;
        CHECK(tp == tp2);

        logitPtrs[dp] = (int64_t) logits[i].getData();

        int64_t mask_offset = (uint8_t*) entries[i].out_mask_pointer - (uint8_t*) masksData;
        CHECK(mask_offset > 0);
        CHECK(mask_offset <= masks_size - mask_stride);
        CHECK(mask_offset % 4 == 0);

        masksOffsets[dp] = mask_offset;
        temperatures[dp] = entries[i].temperature;

        if (mask_offset > max_offset)
            max_offset = mask_offset;

        dp++;

        // auto tokens_vec = tokens[i].get()[0];
        // msg += "\nreqId=" + std::to_string(reqIds[i]) + " ptr=" + std::to_string(logitPtrs[i])
        //     + " n_tokens=" + std::to_string(tokens_vec.size()) + " last=" + std::to_string(tokens_vec.back());
    }

    if (dp > 0)
    {
        cudaMemcpyAsync(cudaMasksData, masksData, max_offset + mask_stride, cudaMemcpyHostToDevice, stream);
        mask_logits_ext(cudaLogitPtrs, cudaMasksOffsets, mask_fractions_offset, temperatures_offset, dp, nVocab,
            mask_stride / 4, tp, stream);
        cudaMemcpyAsync((uint8_t*) masksData + mask_fractions_offset, (uint8_t*) cudaMasksData + mask_fractions_offset,
            dp * sizeof(float), cudaMemcpyDeviceToHost, stream);

#if 0
        for (int lp = 0; lp < (int) logits.size(); lp++)
        {
            tle::Tensor t = logits[lp].copyToCpu(streamPtr);
            streamPtr->synchronize();
            int xlen = (int) entries[lp].num_new_tokens;
            TLLM_LOG_WARNING("temperature = %f at %d len=%d", temperatures[0], lp, xlen);
            if (t.getDataType() == tle::DataType::kFP32)
            {
                float* dp = (float*) t.getData();
                float small = -3e38;
                int numPrint = 0;
                for (int i = 0; i < nVocab; i++)
                {
                    if (dp[i] > small)
                    {
                        if (numPrint < 10)
                        {
                            TLLM_LOG_WARNING("p[%d] = %f at %d", i, dp[i], xlen);
                            numPrint++;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }
            else
            {
                TLLM_LOG_WARNING("not float");
            }
        }
#endif
    }

    uint64_t t3 = monotimer();

    msg += " copy=" + std::to_string(t3 - t2);

    // msg += " reqs=" + std::to_string(reqIds.size()) + " logits=" + std::to_string(logits.size()) + " tokens=" +
    // std::to_string(tokens.size()) + " clients=" + std::to_string(clientIds.size());
    TLLM_LOG_INFO("%s", msg.c_str());
    lastSyncTime = start;
}

void _tlc_set_logits_post_processor(TlcInitParams const* params, tle::ExecutorConfig* config)
{
    if (params->logits_post_processor)
    {
        assert(logits_post_processor == nullptr);

        TLLM_LOG_INFO("Setting logits post processor");

        logits_post_processor = params->logits_post_processor;

        auto logitsConfig = tle::LogitsPostProcessorConfig();
        logitsConfig.setProcessorBatched(logitsPostProcessorFn);
        logitsConfig.setReplicate(false);
        config->setLogitsPostProcessorConfig(logitsConfig);
    }
}
