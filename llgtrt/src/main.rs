use std::env;

use clap::Parser;

use llgtrt::config::CliConfig;
use llgtrt::startup;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = CliConfig::parse();

    if config.debug {
        env::set_var("RUST_LOG", "debug,tokenizers=error");
        // lots of not very useful debug messages
        // env::set_var("TLLM_LOG_LEVEL", "DEBUG");
    } else if config.quiet {
        env::set_var("RUST_LOG", "warn,tokenizers=error");
        env::set_var("TLLM_LOG_LEVEL", "WARNING");
    } else {
        if env::var("RUST_LOG").unwrap_or_default().is_empty() {
            env::set_var("RUST_LOG", "info,tokenizers=error");
        }
        if env::var("TLLM_LOG_LEVEL").unwrap_or_default().is_empty() {
            // don't do that, trtllm is printing "Set logger level to INFO"
            // for every request when running with mpirun
            // env::set_var("TLLM_LOG_LEVEL", "INFO");
        }
    }   

    llgtrt::logging::init_log(llgtrt::logging::LogMode::Normal)?;

    log::info!(
        "logging setup: RUST_LOG={} TLLM_LOG_LEVEL={}",
        env::var("RUST_LOG").unwrap_or_default(),
        env::var("TLLM_LOG_LEVEL").unwrap_or_default()
    );

    log::info!("Setting NCCL_P2P_LEVEL=SYS; see https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.16.0");
    env::set_var("NCCL_P2P_LEVEL", "SYS");

    log::info!("Setting TOKENIZERS_PARALLELISM=false");
    env::set_var("TOKENIZERS_PARALLELISM", "false");

    startup::run_server(config).await
}
