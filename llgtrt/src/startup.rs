use std::env;
use std::fmt::Debug;
use std::sync::Arc;

use anyhow::ensure;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use axum::Router;
use toktrie::{TokEnv, TokEnvWithTrie};
use trtllm_rs::{ClientReqId, ExecutorInit, RequestInit, RequestParams};

use crate::async_exec::AsyncExecutor;
use crate::chat::ChatBuilder;
use crate::config::{ChatTemplates, Config, TrtLlmRuntimeConfig};
use crate::constraint_mgr::ConstraintMgr;
use crate::routes;
use crate::state::AppState;

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

fn load_config_file<T>(
    name0: &Option<String>,
    default_name: String,
    cli_args: serde_json::Value,
) -> anyhow::Result<T>
where
    T: serde::de::DeserializeOwned + Debug,
{
    let name = name0.as_ref().unwrap_or(&default_name);
    let mut json: serde_json::Value = if name0.is_some() || std::fs::exists(name).unwrap_or(false) {
        log::info!("Loading config from {}", name);
        let s = std::fs::read_to_string(name)?;
        serde_json::from_str(&s)?
    } else {
        log::info!("Config file not found, using defaults");
        serde_json::json!({})
    };

    for (k, v) in cli_args.as_object().unwrap() {
        if v.is_null() {
            continue;
        }
        json.as_object_mut().unwrap().insert(k.clone(), v.clone());
    }

    let r = serde_json::from_value(json)
        .map_err(|e| anyhow::anyhow!("Error parsing config file {}: {}", name, e))?;
    log::info!("Loaded config: {:?}", r);
    Ok(r)
}

pub async fn run_server(cli_config: Config) -> anyhow::Result<()> {
    let tokenizer = cli_config
        .tokenizer
        .unwrap_or_else(|| format!("{}/tokenizer.json", cli_config.engine));
    log::info!("Loading tokenizer from {:?}", tokenizer);
    let tok_env = toktrie_hf_tokenizers::ByteTokenizerEnv::from_name(&tokenizer, None)?;
    let tok_env: TokEnv = Arc::new(tok_env);

    let chat_config: ChatTemplates = load_config_file(
        &cli_config.chat_config,
        format!("{}/chat.json", cli_config.engine),
        serde_json::to_value(&cli_config.chat_config_inline)?,
    )?;

    let mut tok_eos = None;
    if let Some(s) = &chat_config.eos_token {
        let toks = tok_env.tokenize(s);
        ensure!(toks.len() == 1, "EOS token must tokenize to a single token");
        tok_eos = Some(toks[0]);
    }

    let mut tok_bos = None;
    if let Some(s) = &chat_config.bos_token {
        let toks = tok_env.tokenize(s);
        ensure!(toks.len() == 1, "BOS token must tokenize to a single token");
        tok_bos = Some(toks[0]);
    }

    let mut exec_config = ExecutorInit {
        engine_path: cli_config.engine.clone(),
        logits_callback: None,
        trt_params: Default::default(),
    };
    let chat_builder = ChatBuilder::new(chat_config.chat_template.as_ref().map(|x| x.as_str()))?;

    let runtime_config: TrtLlmRuntimeConfig = load_config_file(
        &cli_config.runtime_config,
        format!("{}/runtime.json", cli_config.engine),
        serde_json::to_value(&cli_config.runtime_config_inline)?,
    )?;

    let llg_config: serde_json::Value = load_config_file(
        &cli_config.llguidance_config,
        format!("{}/llguidance.json", cli_config.engine),
        serde_json::json!({}),
    )?;

    let p = &mut exec_config.trt_params;

    macro_rules! set_field {
        ($fld:ident) => {
            if let Some(v) = runtime_config.$fld {
                p.$fld = v
                    .try_into()
                    .expect(concat!("Invalid value for ", stringify!($fld)));
            }
        };
    }

    // we default these to true
    p.enable_chunked_context = true;
    p.enable_kv_cache_reuse = true;

    set_field!(enable_chunked_context);
    set_field!(enable_kv_cache_reuse);
    set_field!(max_batch_size);
    set_field!(max_num_tokens);
    set_field!(max_queue_size);
    set_field!(guaranteed_no_evict);
    set_field!(kv_cache_free_gpu_mem_fraction);

    if let Some(v) = runtime_config.kv_cache_host_memory_megabytes {
        p.kv_cache_host_memory_bytes = v * 1024 * 1024;
    }

    log::info!("Initializing executor with config: {:?}", exec_config);
    let trie = tok_env.tok_trie();
    let chat_trie = trie.with_eos_token(tok_eos.unwrap_or(trie.eos_token()));
    let chat_tok_env = Arc::new(TokEnvWithTrie::new(tok_env.clone(), chat_trie));
    let tok_env: TokEnv = chat_tok_env.clone(); // TODO?
    let executor = AsyncExecutor::new(tok_env.clone(), exec_config)?;
    let constraint_mgr = ConstraintMgr::new(tok_env.clone(), chat_tok_env.clone(), llg_config)?;

    let mpi0 = env::var("OMPI_COMM_WORLD_RANK")
        .or_else(|_| env::var("PMI_RANK"))
        .unwrap_or_else(|_| "0".to_string())
        == "0";

    if !mpi0 {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    AsyncExecutor::set_global(executor);

    // warmup request
    log::info!("Warming up executor");
    let mut resp = tok_env.tokenize("The ultimate answer to life, the universe and everything is");
    let (_, mut rx) = AsyncExecutor::lock().add_request(
        RequestInit {
            tokens: resp.clone(),
            params: RequestParams {
                max_new_tokens: 10,
                ..Default::default()
            },
            client_req_id: ClientReqId::new(1),
            is_run: false,
        },
        vec![],
    )?;
    while let Some(r) = rx.recv().await {
        resp.extend_from_slice(&r.response.tokens);
    }
    log::info!("Warmup: {}", tok_env.tok_trie().tokens_dbg(&resp));

    let tok_bos = if tok_bos.is_some() {
        tok_bos
    } else {
        tok_env.tok_trie().info().tok_bos
    };

    let state = AppState {
        tok_env,
        tok_bos,
        tok_eos_chat: tok_eos,
        tok_eos_completions: tok_eos,
        next_client_req_id: std::sync::atomic::AtomicUsize::new(1000),
        chat_builder,
        constraint_mgr,
    };

    let api_key = cli_config.api_key.clone();

    let app = Router::new()
        .route("/v1/completions", post(routes::route_completions))
        .route("/v1/chat/completions", post(routes::route_chat_completions))
        .route("/health_check", get(routes::route_health_check))
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
