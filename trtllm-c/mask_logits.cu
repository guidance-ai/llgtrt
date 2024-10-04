#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

using namespace tensorrt_llm::common;

#define F32_MAX FLT_MAX
#define F16_MAX 65504.0f
#define BF16_MAX 3.38e38f

template <typename T>
__inline__ __device__ void warpReduceMax(T& val, int& idx)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        T other_val = __shfl_xor_sync(FINAL_MASK, val, mask, 32);
        int other_idx = __shfl_xor_sync(FINAL_MASK, idx, mask, 32);

        if (other_val > val)
        {
            val = other_val;
            idx = other_idx;
        }
    }
}

template <typename T>
__inline__ __device__ void blockReduceMax2(T& val, int& idx, T flt_max)
{
    static __shared__ T shared_val[32];
    static __shared__ int shared_idx[32];

    int lane = threadIdx.x & 0x1f; // in-warp thread index
    int wid = threadIdx.x >> 5;    // warp index

    warpReduceMax(val, idx);

    if (lane == 0)
    {
        shared_val[wid] = val;
        shared_idx[wid] = idx;
    }

    __syncthreads();

    // shared_val[0..blockDim.x/32] have the max value of each warp
    if (threadIdx.x < ((blockDim.x + 31) / 32))
    {
        val = shared_val[lane];
        idx = shared_idx[lane];
    }
    else
    {
        // this is hit when there is less than 1024 threads
        val = -flt_max;
        idx = -1;
    }

    // Perform warp-level reduction again across the remaining values
    warpReduceMax(val, idx);
}

template <typename T>
__global__ void mask_logits_kernel(T** logit_ptrs, int64_t* mask_offsets, size_t batch_size, size_t n_vocab,
    size_t mask_stride, float* temperatures, T flt_max, float* mask_fractions)
{
    auto const batch_idx = blockIdx.x;
    auto logits_ptr = logit_ptrs[batch_idx];
    uint32_t* mask_ptr = reinterpret_cast<uint32_t*>((uint8_t*) logit_ptrs + mask_offsets[batch_idx]);

    float max_val = -FLT_MAX;
    float max_val_allowed = -FLT_MAX;
    int max_allowed_idx = 0;
    __shared__ float s_max_val, s_max_val_allowed;
    __shared__ int s_max_val_idx_allowed;

    for (int tid = threadIdx.x; tid < n_vocab; tid += blockDim.x)
    {
        auto mask_j = tid / 32;
        uint32_t mask = mask_j < mask_stride ? mask_ptr[mask_j] : 0;
        auto is_allowed = (mask & (1 << (tid % 32))) != 0;

        auto logit = (float) logits_ptr[tid];

        max_val = max(max_val, logit);
        if (is_allowed && logit > max_val_allowed)
        {
            max_val_allowed = logit;
            max_allowed_idx = tid;
        }
    }

    max_val = blockReduceMax<float>((float) max_val);
    blockReduceMax2(max_val_allowed, max_allowed_idx, FLT_MAX);
    if (threadIdx.x == 0)
    {
        s_max_val = max_val;
        s_max_val_allowed = max_val_allowed;
        s_max_val_idx_allowed = max_allowed_idx;
    }
    __syncthreads();

    auto temperature = temperatures[batch_idx];
    auto is_argmax = temperature < 0.0001f;

    float beta = is_argmax ? 1.0f : 1.0f / temperature;
    float sum_val = 0.0f;
    float sum_val_allowed = 0.0f;

    for (int tid = threadIdx.x; tid < n_vocab; tid += blockDim.x)
    {
        auto mask_j = tid / 32;
        uint32_t mask = mask_j < mask_stride ? mask_ptr[mask_j] : 0;
        auto is_allowed = (mask & (1 << (tid % 32))) != 0;

        auto logit = (float) logits_ptr[tid];

        auto exp = __expf((logit - s_max_val) * beta);
        sum_val += exp;
        if (is_allowed)
        {
            sum_val_allowed += exp;
        }

        auto logit_adjusted = -flt_max;
        if (is_allowed)
        {
            if (is_argmax)
            {
                if (tid == s_max_val_idx_allowed)
                {
                    logit_adjusted = logit;
                }
            }
            else
            {
                logit_adjusted = (logit - s_max_val_allowed) * beta;
            }
        }

        logits_ptr[tid] = (T) logit_adjusted;
    }

    sum_val = blockReduceSum<float>(sum_val);
    sum_val_allowed = blockReduceSum<float>(sum_val_allowed);

    if (threadIdx.x == 0)
    {
        mask_fractions[batch_idx] = sum_val_allowed / sum_val;
    }
}

void mask_logits_ext(int64_t* d_logit_ptrs, // in,out [batch_size]
    int64_t* d_mask_offsets,                // in [int32_t,mask_stride], [batch_size]
    int64_t mask_fractions_offset,          // out, float, [batch_size]
    int64_t temperature_offset,             // in, float, [batch_size]; can be 0.0f for argmax
    size_t batch_size,                      // current batch size
    size_t n_vocab,                         // vocab size
    size_t mask_stride,                     // n_vocab / 32 or thereabouts
    cudaDataType tp,                        // type of logits
    cudaStream_t stream                     // stream for kernel execution
)
{
    assert(n_vocab % 32 == 0);
    dim3 grid(batch_size);
    dim3 block(min((int) n_vocab, 1024));

    float* mask_fractions = reinterpret_cast<float*>((uint8_t*) d_logit_ptrs + mask_fractions_offset);
    float* temperatures = reinterpret_cast<float*>((uint8_t*) d_logit_ptrs + temperature_offset);

#define LAUNCH_KERNEL(T, m)                                                                                            \
    mask_logits_kernel<T><<<grid, block, 0, stream>>>(                                                                 \
        (T**) d_logit_ptrs, d_mask_offsets, batch_size, n_vocab, mask_stride, temperatures, m, mask_fractions)

    switch (tp)
    {
    case CUDA_R_32F: LAUNCH_KERNEL(float, F32_MAX); break;
    case CUDA_R_16BF: LAUNCH_KERNEL(__nv_bfloat16, BF16_MAX); break;
    case CUDA_R_16F: LAUNCH_KERNEL(half, F16_MAX); break;
    default: abort();
    }
}
