use std::sync::Arc;

use anyhow::{anyhow, ensure};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use llguidance::earley::SlicedBiasComputer;
use llguidance::ParserFactory;
use toktrie::InferenceCapabilities;
use trtllm_rs::{ClientReqId, ExecutorInit, RequestInit, RequestParams};

use crate::async_exec::AsyncExecutor;
use crate::chat::ChatParams;
use crate::config::{config_info, CliConfig, LlgTrtConfig};
use crate::jsonutil::json5_to_string;
use crate::lora::LoraCache;
use crate::routes::openai::{ChatCompletionMessageContentPart, ChatCompletionMessageParams};
use crate::state::AppState;
use crate::{jsonutil, py, routes};

async fn auth_middleware(
    req: Request<Body>,
    next: Next,
    api_key: Option<String>,
) -> Result<Response, StatusCode> {
    if let Some(ref key) = api_key {
        if let Some(auth_header) = req.headers().get("Authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if auth_str == format!("Bearer {}", key) {
                    return Ok(next.run(req).await);
                }
            }
        }
        Err(StatusCode::UNAUTHORIZED)
    } else {
        Ok(next.run(req).await)
    }
}

pub async fn run_server(mut cli_config: CliConfig) -> anyhow::Result<()> {
    let mut exec_config = ExecutorInit {
        engine_path: cli_config.engine.clone(),
        logits_callback: None,
        trt_params: Default::default(),
    };

    let defl_config_path = format!("{}/llgtrt.json5", cli_config.engine);
    if cli_config.config.is_empty() {
        if std::fs::exists(&defl_config_path).unwrap_or(false) {
            log::info!("Using default config file {}", defl_config_path);
            cli_config.config.push(defl_config_path);
        } else {
            log::info!(
                "No config files specified and default config file {} not found",
                defl_config_path
            );
        }
    }

    let mut config = LlgTrtConfig::default();

    if cli_config.print_config {
        log::info!("Skipping tokenizer config load");
    } else {
        let tokenizer_folder = cli_config.tokenizer.as_ref().unwrap_or(&cli_config.engine);
        let tokenizer_config = format!("{}/tokenizer_config.json", tokenizer_folder);
        log::info!("Loading tokenizer config from {:?}", tokenizer_config);
        config.tokenizer = serde_json::from_reader(std::fs::File::open(tokenizer_config)?)
            .map_err(|e| anyhow!("error loading tokenizer_config.json: {}", e))?;
    }

    let mut config = serde_json::to_value(&config)?;

    for file_name in &cli_config.config {
        log::info!("Loading JSON5 config from {:?}", file_name);
        let file_content = std::fs::read_to_string(&file_name)
            .map_err(|e| anyhow!("Error reading config file {}: {}", file_name, e))?;
        let mut patch = json5::from_str::<serde_json::Value>(&file_content)
            .map_err(|e| anyhow!("Error in JSON5 in {}: {}", file_name, e))?;
        if let Some(p) = patch["py"]["input_processor"].as_str() {
            let p = std::path::Path::new(p);
            if p.is_relative() {
                let p = std::path::Path::new(file_name).parent().unwrap().join(p);
                patch["py"]["input_processor"] =
                    serde_json::Value::String(p.to_str().unwrap().to_string());
            }
        }
        jsonutil::json_merge(&mut config, &patch);
    }

    let mut config: LlgTrtConfig =
        serde_json::from_value(config).map_err(|e| anyhow!("Error interpreting config: {}", e))?;

    if cli_config.debug_llg {
        config.llguidance.log_level = 2;
    }

    if cli_config.print_config {
        log::info!("Skipping separate chat template load");
    } else {
        let chat_template = cli_config
            .chat_template
            .clone()
            .unwrap_or_else(|| format!("{}/chat_template.j2", cli_config.engine));
        log::info!("Checking for separate chat template in {:?}", chat_template);
        if std::fs::exists(&chat_template)? {
            config.tokenizer.chat_template = Some(std::fs::read_to_string(chat_template)?);
        }
    }

    if cli_config.print_config || cli_config.print_complete_config {
        let r = json5_to_string(
            &serde_json::to_value(&config)?,
            &serde_json::to_value(&LlgTrtConfig::default())?,
            &config_info(),
        );
        log::info!("Printing merged config to stdout");
        println!("{}", r);
        return Ok(());
    }

    if cli_config.print_chat_template {
        log::info!("Printing chat template to stdout");
        if config.tokenizer.chat_template.is_none() {
            log::warn!("No chat template found");
            return Ok(());
        }
        print!("{}", config.tokenizer.chat_template.as_ref().unwrap());
        return Ok(());
    }

    let runtime_config = &config.runtime;
    let p = &mut exec_config.trt_params;

    macro_rules! set_field {
        ($fld:ident) => {
            p.$fld = runtime_config
                .$fld
                .try_into()
                .expect(concat!("Invalid value for ", stringify!($fld)));
        };
    }

    set_field!(enable_chunked_context);
    set_field!(enable_kv_cache_reuse);
    set_field!(enable_batch_size_tuning);
    set_field!(enable_max_num_tokens_tuning);
    set_field!(max_batch_size);
    set_field!(max_num_tokens);
    set_field!(max_queue_size);
    set_field!(guaranteed_no_evict);
    set_field!(kv_cache_free_gpu_mem_fraction);
    p.kv_cache_host_memory_bytes = runtime_config.kv_cache_host_memory_megabytes * 1024 * 1024;

    log::info!("Initializing executor with config: {:?}", exec_config);

    let py_state = py::init(&cli_config, &config)?;
    if false {
        let r = py_state.run_input_processor(ChatParams {
            messages: &vec![ChatCompletionMessageParams::User {
                content: ChatCompletionMessageContentPart::Text("Hello world!".to_string()),
                name: None,
            }],
            tools: &vec![],
            json_schema: None,
        })?;
        log::warn!("early stop {r:?}");
        return Ok(());
    }

    let (executor, tok_env, chat_builder) = AsyncExecutor::new(&cli_config, &config, exec_config)?;

    // we only get here on rank 0

    let mut parser_factory = ParserFactory::new(
        &tok_env,
        InferenceCapabilities {
            ff_tokens: false, // not supported yet
            backtrack: false, // unlikely
            ..Default::default()
        },
        &SlicedBiasComputer::general_slices(),
    )
    .expect("Error creating parser factory");
    *parser_factory.limits_mut() = config.llguidance.limits.clone();
    parser_factory.set_stderr_log_level(config.llguidance.log_level);

    if let Some(t) = config.tokenizer.json_start_token.as_ref() {
        ensure!(
            tok_env.tok_trie().get_special_token(t).is_some(),
            "json_start_token {:?} not found in tokenizer",
            t
        )
    }

    AsyncExecutor::set_global(executor);

    let trie = tok_env.tok_trie();

    let state = AppState {
        tok_bos: trie.info().tok_bos,
        tok_eos_chat: Some(trie.info().tok_eos),
        tok_eos_completions: Some(trie.info().tok_eos),
        json_start_token_name: config.tokenizer.json_start_token.clone(),
        tok_env,
        next_client_req_id: std::sync::atomic::AtomicUsize::new(1000),
        chat_builder,
        parser_factory,
        lora_root: cli_config.lora_root,
        lora_cache: LoraCache::new(),
        py_state,
    };

    // warmup request
    log::info!("Warming up executor");
    let mut warmup_tokens =
        state.tokenize_with_bos("The ultimate answer to life, the universe and everything is");
    log::debug!("Warmup tokens: {:?}", warmup_tokens);
    let (_, mut rx) = AsyncExecutor::lock().add_request(
        &RequestInit {
            tokens: warmup_tokens.clone(),
            params: RequestParams {
                max_new_tokens: 10,
                ..Default::default()
            },
            client_req_id: ClientReqId::new(1),
            lora_params: None,
            is_run: false,
        },
        vec![],
    )?;
    while let Some(r) = rx.recv().await {
        warmup_tokens.extend_from_slice(&r.response.tokens);
    }
    log::info!(
        "Warmup: {}",
        state.tok_env.tok_trie().tokens_dbg(&warmup_tokens)
    );

    let api_key = cli_config.api_key.clone();

    let app = Router::new()
        .route("/v1/completions", post(routes::route_completions))
        .route("/v1/chat/completions", post(routes::route_chat_completions))
        .route("/v1/health/live", get(routes::live_check))
        .route("/v1/health/model", get(routes::model_check))
        .route("/v1/health/ready", get(routes::ready_check))
        .route("/v1/run", post(routes::route_llguidance))
        .route("/guidance", post(routes::route_llguidance))
        .with_state(Arc::new(state))
        .layer(middleware::from_fn(move |req, next| {
            auth_middleware(req, next, api_key.clone())
        }));

    let address = format!("{}:{}", cli_config.host, cli_config.port);
    // make sure we (almost) always print this
    log::warn!("Starting server at {}", address);

    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
}
