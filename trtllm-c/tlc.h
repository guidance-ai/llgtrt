#ifndef TLC_H
#define TLC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef uint64_t TlcReqId;
    typedef uint64_t TlcClientId;
    typedef char* TlcStatus;

    typedef struct
    {
        TlcReqId _req_id;
        TlcClientId _client_req_id;
        // this all tokens, both prompt and completion
        int32_t const* _tokens;
        uint32_t _num_tokens;
        // set by the callback (initially 1.0)
        float temperature;
        // set by the callback (initially NULL)
        uint32_t* out_mask_pointer;
    } TlcLogitsEntry;

    typedef void (*TlcLogitsPostProcessor)(TlcLogitsEntry* logits, uint32_t num_logits);

    typedef struct
    {
        // TODO peft

        // defaults to false (max_utilization); set to true to guaranteed_no_evict
        bool guaranteed_no_evict;

        // defaults to 1000
        int32_t iter_stats_max_iterations;
        // defaults to 0
        int32_t request_stats_max_iterations;

        // defaults to 128
        int32_t max_batch_size;
        // defaults to 8192
        int32_t max_num_tokens;

        // The maximum number of requests allowed in queue before rejecting new requests.
        // defaults to 0
        int32_t max_queue_size;

        // The maximum time in microseconds a scheduled request can remain idle before getting terminated.
        // defaults to 3 minutes
        uint64_t max_queue_delay_microseconds;

        // default to false
        bool enable_chunked_context;

        // defaults to 1.0
        float gpu_weights_percent;

        // KV cache config
        // defaults to 0.9
        float kv_cache_free_gpu_mem_fraction;
        // defaults to kv_cache_free_gpu_mem_fraction
        int32_t max_tokens_in_paged_kv_cache;
        // defaults to 0
        size_t kv_cache_host_memory_bytes;
        // defaults to true
        bool kv_cache_onboard_blocks;
        // defaults to 0 (disabled)
        int32_t max_attention_window_size;
        // when set to 0, use default
        int32_t sink_token_length;
        // defaults to false (prefix caching)
        bool enable_kv_cache_reuse;

        // both default to false
        bool enable_batch_size_tuning;
        bool enable_max_num_tokens_tuning;
    } TlcEngineParams;

    typedef struct
    {
        char const* engine_path;
        TlcLogitsPostProcessor logits_post_processor;
        TlcEngineParams engine_params;
    } TlcInitParams;

    // TensorRT tensor class support
    typedef struct
    {
        int64_t const* dims_ptr;
        size_t num_dims;
    } TlcShape;

    typedef struct
    {
        int32_t data_type;
        void const* data_ptr;
        TlcShape shape;
    } TlcTensor;

    typedef struct
    {
        uint64_t lora_id;
        TlcTensor weights;
        TlcTensor config;
    } TlcLoraParams;

    typedef struct
    {
        bool use_logits_post_processor;
        bool streaming;
        bool logprobs;
        uint32_t max_new_tokens;
        uint32_t num_return_sequences;
        uint32_t eos_token_id;
        float temperature;
        float top_p;
        float frequency_penalty;
        float presence_penalty;
        float priority;
        uint32_t top_k;
        uint32_t min_tokens;
        uint64_t seed;
    } TlcRequestParams;

    typedef struct
    {
        // PromptTuningConfig
        TlcTensor prompt_table;
        TlcTensor prompt_tasks; // vec<u64>

        // MropeConfig
        TlcTensor mrope_rotary_sin_cos;
        int32_t mrope_position_deltas;

        TlcTensor skip_cross_attn_blocks;

        TlcTensor encoder_input_features;
        int32_t encoder_output_length;
        TlcTensor cross_attention_masks;

        TlcTensor input_position_ids; // vec<u32>
    } TlcPromptParams;

    typedef struct
    {
        int32_t* tokens;
        uint32_t num_tokens;
        TlcClientId client_req_id;
        TlcRequestParams params;
        TlcLoraParams lora_params;
        TlcPromptParams prompt_params;
    } TlcRequest;

    /// @brief The reason why the model stopped generating tokens for a request.
    typedef enum
    {
        /// @brief The request is not finished.
        FINISH_REASON_NOT_FINISHED = 0,

        /// @brief The request finished because the end id was generated.
        FINISH_REASON_END_ID = 1,

        /// @brief The request finished because a stop word was generated.
        FINISH_REASON_STOP_WORDS = 2,

        /// @brief The request finished because the maximum number of tokens was reached.
        FINISH_REASON_LENGTH = 3,
    } TlcFinishReason;

    typedef struct
    {
        TlcReqId req_id;
        uint32_t sequence_idx; // when num_return_sequences > 1
        bool is_seq_final;
        bool is_req_final;
        TlcFinishReason finish_reason;

        char const* error;

        // these are only output tokens
        uint32_t num_tokens;
        int32_t const* tokens;
        uint32_t num_logprobs;
        float const* logprobs;
    } TlcResponse;

    typedef struct TlcExecutor TlcExecutor;

    void tlc_default_init_params(TlcInitParams* params);
    TlcStatus tlc_init(TlcInitParams const* params, TlcExecutor** res);
    bool tlc_can_enqueue_request(TlcExecutor* ctx);
    void tlc_shutdown(TlcExecutor* ctx);
    TlcStatus tlc_enqueue_request(TlcExecutor* ctx, TlcRequest const* request, TlcReqId* res);
    TlcStatus tlc_cancel_request(TlcExecutor* ctx, TlcReqId req_id);
    TlcStatus tlc_await_responses(
        TlcExecutor* ctx, uint32_t timeout_ms, TlcResponse const** responses, uint32_t* num_responses);
    void* tlc_alloc_logit_data(int32_t mask_stride_, int32_t max_batch_size_);
    float* tlc_mask_fraction_ptr(void);

#ifdef __cplusplus
}
#endif

#endif // TLC_H