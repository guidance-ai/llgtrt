use clap::Parser;
use llguidance::api::ParserLimits;
use serde::{Deserialize, Serialize};

use crate::tokenizer::TokenizerConfig;

const CONFIG_INFO: &str = include_str!("config_info.json");
pub fn config_info() -> serde_json::Value {
    serde_json::from_str(CONFIG_INFO).unwrap()
}

const CONFIG_OPTIONS: &str = "Configuration files handling";

#[derive(Debug, Serialize, Deserialize)]
pub struct TrtLlmRuntimeConfig {
    /// Make the scheduler more conservative, so that a started request is never evicted.
    /// Defaults to false (which improves throughput)
    pub guaranteed_no_evict: bool,

    /// Maximum number of concurrent requests
    pub max_batch_size: usize,

    /// Maximum number of tokens in batch
    pub max_num_tokens: usize,

    /// Maximum number of requests in queue (when batch already full)
    pub max_queue_size: usize,

    /// Chunk prefill/generation into pieces
    /// Defaults to true (unlike trtllm)
    pub enable_chunked_context: bool,

    /// Prefix-caching (LRU-reuse blocks between requests)
    /// Defaults to true (unlike trtllm)
    pub enable_kv_cache_reuse: bool,

    /// Fraction of free GPU memory to use for KV cache
    pub kv_cache_free_gpu_mem_fraction: f32,

    /// Host memory to use for KV cache
    pub kv_cache_host_memory_megabytes: usize,

    /// Controls whether offloaded blocks should be onboarded back into primary memory before being reused.
    /// Defaults to true.
    pub kv_cache_onboard_blocks: bool,

    /// The fraction of the KV Cache memory should be reserved for cross attention
    /// If set to p, self attention will use 1-p of KV Cache memory and cross attention
    /// will use p of KV Cache memory.
    /// Should only be set when using encoder-decoder model.
    pub cross_kv_cache_fraction: Option<f32>,

    /// Only blocks with priority > mSecondaryOfflineMinPriority can be offloaded to secondary memory.
    pub secondary_offload_min_priority: Option<i32>,

    /// Max size of the KV cache event buffer
    pub event_buffer_max_size: Option<usize>,

    /// Control automatic tuning of batch size
    /// Defaults to true (unlike trtllm)
    pub enable_batch_size_tuning: bool,

    /// Control automatic tuning of max num tokens
    /// Defaults to true (unlike trtllm)
    pub enable_max_num_tokens_tuning: bool,
}

impl Default for TrtLlmRuntimeConfig {
    fn default() -> Self {
        Self {
            guaranteed_no_evict: false,
            max_batch_size: 128,
            max_num_tokens: 8192,
            max_queue_size: 0,
            enable_chunked_context: true,
            enable_kv_cache_reuse: true,
            kv_cache_free_gpu_mem_fraction: 0.9,
            kv_cache_host_memory_megabytes: 0,
            enable_batch_size_tuning: true,
            enable_max_num_tokens_tuning: true,
            kv_cache_onboard_blocks: true,
            cross_kv_cache_fraction: None,
            secondary_offload_min_priority: None,
            event_buffer_max_size: None,
            n_draft_tokens: None
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct LlgTrtPyConfig {
    /// Path to a Python script that does multi-modal and chat template processing
    /// Defaults to <engine>/input_processor.py if it exists
    pub input_processor: Option<String>,

    /// Path to HuggingFace model directory; defaults to engine directory
    pub hf_model_dir: Option<String>,

    /// Path to TRT-LLM visual engine; defaults to <engine>/visual_engine
    pub visual_engine_dir: Option<String>,

    /// Additional arguments passed to the Python script
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct LlgTrtConfig {
    /// TensorRT-LLM runtime parameters
    /// Defaults should be reasonable, otherwise see
    /// https://nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html
    pub runtime: TrtLlmRuntimeConfig,

    /// Tokenizer configuration (defaults to tokenizer_config.json contents)
    /// Typically no changes are needed here, except for chat_template
    /// which is best overridden with --chat-template filename.j2 option.
    pub tokenizer: TokenizerConfig,

    /// Configuration for the LLGuidance constraint library
    pub llguidance: LlgConfig,

    /// Configuration for the embedded Python API
    pub py: LlgTrtPyConfig,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LlgConfig {
    /// Override any of the parser limits.
    pub limits: ParserLimits,

    /// Log level which goes to stderr. In-memory logs per-sequence are managed by ConstraintInit.log_level.
    pub log_level: u32,
}

impl Default for LlgConfig {
    fn default() -> Self {
        Self {
            limits: ParserLimits::default(),
            log_level: 1,
        }
    }
}

#[derive(Parser, Debug, Serialize, Deserialize)]
pub struct CliConfig {
    /// Host to bind to
    #[arg(long, short = 'H', default_value_t = String::from("0.0.0.0"))]
    pub host: String,

    /// Port to bind to
    #[arg(long, short, default_value_t = 3000)]
    pub port: usize,

    /// Path to a compiled TensorRT-LLM engine
    #[arg(long, short = 'E')]
    pub engine: String,

    /// Path to folder with HF tokenizer.json and tokenizer_config.json files; defaults to --engine
    #[arg(long, short = 'T')]
    pub tokenizer: Option<String>,

    /// Path to a compiled TensorRT-LLM draft engine
    #[arg(long)]
    pub draft_engine: Option<String>,

    /// Path to folder with HF tokenizer.json and tokenizer_config.json files for draft engine; defaults to ---draft-engine
    #[arg(long)]
    pub draft_tokenizer: Option<String>,

    /// Debug output
    #[arg(long, short = 'd')]
    pub debug: bool,

    /// Debug output from llguidance
    #[arg(long, short = 'D')]
    pub debug_llg: bool,

    /// Quiet output (only warnings)
    #[arg(long, short = 'q')]
    pub quiet: bool,

    /// Api Key to access the server
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Path to JSON5 configuration file; multiple files are JSON-merged in order; defaults to:
    /// <engine>/llgtrt.json5 if it exists
    #[arg(long, short = 'C', help_heading = CONFIG_OPTIONS)]
    pub config: Vec<String>,

    /// Path to chat template file; defaults to <engine>/chat_template.j2 if it exists
    /// Overrides values in all configs.
    #[arg(long, help_heading = CONFIG_OPTIONS)]
    pub chat_template: Option<String>,

    /// Root directory for LoRA weights; defaults to None (LoRA disabled)
    #[arg(long, help_heading = CONFIG_OPTIONS)]
    pub lora_root: Option<String>,

    /// Print the merged configuration and exit
    #[arg(long, help_heading = CONFIG_OPTIONS)]
    pub print_config: bool,

    /// Similar to --print-config, but includes chat template and tokenizer config
    #[arg(long, help_heading = CONFIG_OPTIONS)]
    pub print_complete_config: bool,

    /// Print the chat template and exit
    #[arg(long, help_heading = CONFIG_OPTIONS)]
    pub print_chat_template: bool,

    /// Test python plugin initialization and exit
    #[arg(long)]
    pub test_py: bool,
}
