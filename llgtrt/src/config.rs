use clap::Parser;
use serde::{Deserialize, Serialize};

use crate::{constraint_mgr::LlgConfig, tokenizer::TokenizerConfig};

const CONFIG_INFO: &str = include_str!("config_info.json");
pub fn config_info() -> serde_json::Value {
    serde_json::from_str(CONFIG_INFO).unwrap()
}

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
        }
    }
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

    /// Path to JSON5 configuration file; multiple files are JSON-merged in order; defaults to:
    /// <engine>/llgtrt.json5 if it exists
    #[arg(long, short = 'C')]
    pub config: Vec<String>,

    /// Path to chat template file; defaults to <engine>/chat_template.j2 if it exists
    /// Overrides values in all configs.
    #[arg(long)]
    pub chat_template: Option<String>,

    /// When present, save the merged configuration to this file and exit; use '-' for stdout
    #[arg(long)]
    pub save_config: Option<String>,

    /// Similar to --save-config, but includes chat template and tokenizer config
    #[arg(long)]
    pub save_complete_config: Option<String>,

    /// Debug output
    #[arg(long, short = 'd')]
    pub debug: bool,

    /// Quiet output (only warnings)
    #[arg(long, short = 'q')]
    pub quiet: bool,

    /// Api Key to access the server
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}
