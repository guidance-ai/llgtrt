#pragma once
#include <cuda_runtime.h>
#include <cstdint>

void mask_logits(int64_t* d_logit_ptrs, int64_t* d_mask_offsets, size_t batch_size, size_t n_vocab, size_t mask_stride,
    cudaDataType tp, cudaStream_t stream);

// all offset fields refer to byte offsets from d_logit_ptrs
void mask_logits_ext(int64_t* d_logit_ptrs, // in,out [batch_size]
    int64_t* d_mask_offsets,                // in [int32_t,mask_stride], [batch_size]
    int64_t mask_fractions_offset,          // out, float, [batch_size]
    int64_t temperature_offset,             // in, float, [batch_size]; can be 0.0f for argmax
    int64_t ln_min_p_offset,                // in, float, [batch_size]; log_e(min_p) for min_p > 0.0f, -FLT_MAX otherwise
    size_t batch_size,                      // current batch size
    size_t n_vocab,                         // vocab size
    size_t mask_stride,                     // n_vocab / 32 or thereabouts
    cudaDataType tp,                        // type of logits
    cudaStream_t stream                     // stream for kernel execution
);
