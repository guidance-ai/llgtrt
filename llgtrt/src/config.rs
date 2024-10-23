use clap::{Args, Parser};
use serde::{Deserialize, Serialize};

const TRT_CONFIG: &str = "TensorRT-LLM runtime config (runtime.json)";

#[derive(Args, Debug, Serialize, Deserialize)]
pub struct TrtLlmRuntimeConfig {
    /// When set to true, the scheduler is more conservative, so that a started request is never evicted; defaults to false (which improves throughput)
    #[clap(long, help_heading = TRT_CONFIG)]
    pub guaranteed_no_evict: Option<bool>,

    /// Maximum number of concurrent requests; defaults to 128
    #[clap(long, help_heading = TRT_CONFIG)]
    pub max_batch_size: Option<usize>,

    /// Maximum number of tokens in batch; defaults to 8192
    #[clap(long, help_heading = TRT_CONFIG)]
    pub max_num_tokens: Option<usize>,

    /// Maximum number of requests in queue (when batch already full); defaults to 0
    #[clap(long, help_heading = TRT_CONFIG)]
    pub max_queue_size: Option<usize>,

    /// Chunk prefill/generation into pieces; defaults to true (unlike trtllm)
    #[clap(long, help_heading = TRT_CONFIG)]
    pub enable_chunked_context: Option<bool>,

    /// Prefix-caching (LRU-reuse blocks between requests); defaults to true (unlike trtllm)
    #[clap(long, help_heading = TRT_CONFIG)]
    pub enable_kv_cache_reuse: Option<bool>,

    /// Fraction of free GPU memory to use for KV cache; defaults to 0.9
    #[clap(long, help_heading = TRT_CONFIG)]
    pub kv_cache_free_gpu_mem_fraction: Option<f32>,

    /// Host memory to use for KV cache; defaults to 0
    #[clap(long, help_heading = TRT_CONFIG)]
    pub kv_cache_host_memory_megabytes: Option<usize>,
}

#[derive(Parser, Debug, Serialize, Deserialize)]
pub struct Config {
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

    /// Path to JSON file TensorRT-LLM runtime config; defaults to runtime.json in engine dir
    #[arg(long, short = 'R')]
    pub runtime_config: Option<String>,

    /// Path to JSON file with llguidance library config; defaults to llguidance.json in engine dir
    #[arg(long, short = 'L')]
    pub llguidance_config: Option<String>,

    /// Debug output
    #[arg(long, short = 'd')]
    pub debug: bool,

    /// Quiet output (only warnings)
    #[arg(long, short = 'q')]
    pub quiet: bool,

    #[clap(flatten)]
    pub runtime_config_inline: TrtLlmRuntimeConfig,

    /// Api Key to access the server
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}
