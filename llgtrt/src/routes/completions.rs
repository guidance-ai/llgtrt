//! https://platform.openai.com/docs/api-reference/completions/create
use anyhow::{anyhow, bail, ensure, Error, Result};
use async_stream::try_stream;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use core::panic;
use std::cmp::min;
use futures_core::Stream;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use llguidance::Constraint;
use safetensors::Dtype;
use serde_json::{json, Value};
use std::fmt::Display;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedReceiver;
use toktrie::TokEnv;
use trtllm_rs::{ClientReqId, DraftParams, FinishReason, LoraParams, ReqId, RequestInit, RequestParams, Tensor, TlcDataType};
use uuid::Uuid;

use crate::async_exec::{map_finish_reason, AsyncExecutor, StepResults};
use crate::chat::ChatParams;
use crate::error::AppError;
use crate::lora::LoraCache;
use crate::py::PyPromptParams;
use crate::routes::api_ext::{tools_to_schema, LlgLogLevel};
use crate::routes::openai::{JsonSchemaOptions, LoadLoraWeightsOption, ResponseFormat, ToolChoice};
use crate::state::AppState;

use super::api_ext::{
    InitialRunResponse, RunForkResponse, RunRequest, RunResponse, RunUsageResponse,
};
use super::openai::{
    ChatCompletion, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionChunkDelta, ChatCompletionCreateParams, ChatCompletionMessage,
    CommonCreateParams, Completion, CompletionCreateParams, LogProbs, Role, ToolChoiceOption,
    TopTokenLogProb, Usage,
};

#[derive(Debug, Clone, Default)]
struct ForkInfo {
    text: Vec<u8>,
    pending_bytes: Vec<u8>,
    logs: String,
    stop_reason: Option<trtllm_rs::FinishReason>,
}

struct ReqInfo {
    req_id: ReqId,
    client_req_id: ClientReqId,
    prompt: String,
    return_expanded_prompt: bool,
    is_chat: bool,
    is_run: bool,
    usage: Usage,
    cmpl_id: String,
    created: u64,
    model_name: String,
    tok_env: TokEnv,
    recv: tokio::sync::mpsc::UnboundedReceiver<StepResults>,
    stop_words: Vec<Vec<u8>>,
    forks: Vec<ForkInfo>,
}

impl Display for ReqInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ReqInfo {{ {} {} {} }}",
            self.req_id, self.client_req_id, self.cmpl_id
        )
    }
}

pub struct RequestInput {
    pub tokens: Vec<u32>,
    pub prompt: String,
    pub prompt_params: Option<Arc<PyPromptParams>>,
}

impl RequestInput {
    pub fn from_text(app_state: &AppState, text: &str) -> Self {
        Self {
            tokens: app_state.tokenize_with_bos(text),
            prompt: text.to_string(),
            prompt_params: None,
        }
    }
}

fn req_params_from_openai(params: &CommonCreateParams) -> Result<RequestParams> {
    ensure!(params.logit_bias.is_none(), "logit_bias not supported yet");
    ensure!(
        params.max_tokens.is_none() || params.max_completion_tokens.is_none(),
        "max_tokens and max_completion_tokens are mutually exclusive"
    );
    ensure!(
        params.logprobs.unwrap_or(0) <= 1,
        "logprobs > 1 not supported yet"
    );
    match &params.response_format {
        Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaOptions { name: Some(n), .. },
        }) => {
            ensure!(
                n.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "response format name must be alphanumeric, underscore, or dash"
            );
            ensure!(
                n.len() <= 64,
                "response format name must be 64 characters or less"
            );
        }
        _ => {}
    }

    let mut r = RequestParams {
        temperature: params.temperature,
        top_p: params.top_p,
        min_p: params.min_p,
        max_new_tokens: params
            .max_completion_tokens
            .unwrap_or_else(|| params.max_tokens.unwrap_or(16)) as u32,
        num_return_sequences: params.n as u32,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        logprobs: params.logprobs.unwrap_or(0) > 0,
        min_tokens: params.min_tokens as u32,
        ..Default::default()
    };

    if let Some(prio) = params.priority {
        r.priority = prio;
    }

    if let Some(seed) = params.seed {
        r.seed = seed;
    } else {
        r.seed = rand::random();
    }

    Ok(r)
}

fn load_lora_tensors<P: AsRef<Path>>(
    base_path: P,
    lora_model: &str,
) -> Result<(Tensor, Tensor), Error> {
    // Construct paths for the safetensors file
    let safetensors_path = base_path
        .as_ref()
        .join(lora_model)
        .with_extension("safetensors");
    log::info!("Loading LoRA weights from {}", safetensors_path.display());

    // Ensure the resulting path is within the lora root hierarchy
    let canonicalized = std::fs::canonicalize(safetensors_path.clone()).map_err(|err| {
        let context = format!(
            "Failed to find LoRA weights file: {}. Error: {}",
            safetensors_path.display(),
            err
        );
        log::error!("{}", context);
        anyhow!(context)
    })?;
    if !canonicalized.starts_with(base_path.as_ref()) {
        return Err(anyhow!(
            "LoRA weights path {} is outside the LoRA base directory {}",
            safetensors_path.display(),
            base_path.as_ref().display()
        ));
    }

    let fp = std::fs::File::open(safetensors_path.clone()).map_err(|err| {
        let context = format!(
            "Failed to open LoRA weights file: {}. Error: {}",
            safetensors_path.display(),
            err
        );
        log::error!("{}", context);
        anyhow!(context)
    })?;
    let content = unsafe { memmap2::MmapOptions::new().map(&fp)? };
    let safetensors = safetensors::SafeTensors::deserialize(&content)?;

    // Read the config array
    let config_view = safetensors.tensor("config")?;
    let config_size: Vec<i64> = config_view.shape().iter().map(|&x| x as i64).collect();
    let config_tensor = Tensor {
        size: config_size,
        dtype: safetensors_dtype_to_tlc(config_view.dtype()),
        data: config_view.data().to_vec(),
    };

    // Read the weights array
    let weights_view = safetensors.tensor("weights")?;
    let weights_size: Vec<i64> = weights_view.shape().iter().map(|&x| x as i64).collect();
    let weights_tensor = Tensor {
        size: weights_size,
        dtype: safetensors_dtype_to_tlc(weights_view.dtype()),
        data: weights_view.data().to_vec(),
    };

    Ok((weights_tensor, config_tensor))
}

fn req_lora_params_from_openai(
    params: &CommonCreateParams,
    lora_root: Option<String>,
    lora_cache: &LoraCache,
    load_lora_weights: bool,
) -> Result<Option<LoraParams>> {
    if let Some(lora_root) = lora_root {
        if let Some(lora_model) = &params.lora_model {
            let base_path: &Path = Path::new(lora_root.as_str());
            if load_lora_weights {
                let (weights, config) = load_lora_tensors(base_path, lora_model)?;
                Ok(Some(LoraParams {
                    lora_id: lora_cache.resolve_id(lora_model),
                    weights: Some(weights),
                    config: Some(config),
                }))
            } else {
                Ok(Some(LoraParams {
                    lora_id: lora_cache.resolve_id(lora_model),
                    weights: None,
                    config: None,
                }))
            }
        } else {
            Ok(None)
        }
    } else {
        if let Some(_lora_model) = &params.lora_model {
            Err(anyhow!("lora_model specified but lora_root not set"))
        } else {
            Ok(None)
        }
    }
}

fn validate_compl(req: &CompletionCreateParams) -> Result<()> {
    ensure!(req.prompt.len() == 1, "prompt must be a single string");
    ensure!(req.suffix.is_none(), "suffix is not supported");
    ensure!(req.best_of == 1, "best_of must be 1 or missing");
    ensure!(req.echo == false, "echo not supported");
    ensure!(req.logprobs.is_none(), "logprobs not supported yet");
    let _ = req_params_from_openai(&req.params)?;
    Ok(())
}

fn validate_chat(req: &ChatCompletionCreateParams) -> Result<()> {
    let _ = req_params_from_openai(&req.params)?;
    ensure!(
        matches!(req.tool_choice, ToolChoice::Simple(_)),
        "only simple options are currently supported for tool_choice"
    );
    ensure!(
        req.tools.is_empty() || req.params.response_format.is_none(),
        "response_format cannot be specified together with tools"
    );
    Ok(())
}

fn llg_grammar(params: &CommonCreateParams) -> Result<Option<TopLevelGrammar>> {
    let grm = match &params.response_format {
        Some(ResponseFormat::Llguidance { grammar }) => grammar.clone(),
        Some(ResponseFormat::JsonObject)
        | Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaOptions { strict: false, .. },
        }) => {
            log::debug!("using generic JSON schema");
            json_to_llg(json!({ "type": "object" }))?
        }
        Some(ResponseFormat::JsonSchema {
            json_schema:
                JsonSchemaOptions {
                    schema: None,
                    strict: true,
                    ..
                },
        }) => {
            bail!("missing schema in strict mode")
        }
        Some(ResponseFormat::JsonSchema {
            json_schema:
                JsonSchemaOptions {
                    schema: Some(schema),
                    strict: true,
                    ..
                },
        }) => {
            log::debug!("using strict JSON schema");
            json_to_llg(schema.clone())?
        }
        Some(ResponseFormat::LarkGrammar { lark_grammar }) => {
            log::debug!("using Lark grammar");
            TopLevelGrammar::from_lark(lark_grammar.clone())
        }
        _ => {
            if params.min_p > 0.0 {
                // Returning a Dummy-grammar to enforce logit processing when min_p is set
                let grm = TopLevelGrammar::from_regex(r"(\n|.)*");
                return Ok(Some(grm));
            }
            return Ok(None);
        }
    };
    Ok(Some(grm))
}

fn is_lora_cache_miss_error(err: &AppError) -> bool {
    let target_substring = "Please send LoRA weights with request";
    err.to_string().contains(target_substring)
}

fn build_request_init(
    tokens: Vec<u32>,
    req_params: RequestParams,
    client_req_id: ClientReqId,
    is_run: bool,
    params: &CommonCreateParams,
    app_state: &AppState,
    load_lora_weights: bool,
) -> Result<RequestInit, AppError> {
    let lora_params = req_lora_params_from_openai(
        params,
        app_state.lora_root.clone(),
        &app_state.lora_cache,
        load_lora_weights,
    )?;

    let request_init = RequestInit {
        tokens,
        params: req_params,
        client_req_id,
        is_run,
        lora_params,
        draft_params: None  // TODO fill out
    };
    Ok(request_init)
}

fn build_req_info(
    req_id: ReqId,
    n_forks: u32,
    client_req_id: ClientReqId,
    cmpl_id: String,
    prompt: &str,
    prompt_tokens: usize,
    params: &CommonCreateParams,
    tok_env: &TokEnv,
    is_chat: bool,
    is_run: bool,
    recv: UnboundedReceiver<StepResults>,
) -> Result<ReqInfo, AppError> {
    let usage = Usage {
        prompt_tokens,
        completion_tokens: 0,
        total_tokens: prompt_tokens,
    };

    let info = ReqInfo {
        req_id,
        client_req_id,
        prompt: prompt.to_string(),
        return_expanded_prompt: params.return_expanded_prompt.unwrap_or(false),
        is_chat,
        is_run,
        cmpl_id,
        model_name: params.model.clone(),
        created: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        recv,
        tok_env: tok_env.clone(),
        forks: vec![ForkInfo::default(); n_forks as usize],
        stop_words: params
            .stop
            .as_ref()
            .map(|stops| stops.iter().map(|s| s.as_bytes().to_vec()).collect())
            .unwrap_or_default(),
        usage,
    };

    Ok(info)
}

async fn completions_stream_or_not(
    stream: bool,
    info: ReqInfo,
    final_content: Option<Vec<u8>>
) -> Result<Response<Body>, AppError> {
    if stream {
        completions_stream(info).await.map(|r| r.into_response())
    } else {
        completions(info, final_content).await.map(|r| r.into_response())
    }
}

async fn mk_req_info(
    app_state: &AppState,
    req_input: RequestInput,
    params: &CommonCreateParams,
    is_chat: bool,
    is_run: bool,
) -> Result<Response, AppError> {
    let mut req_params = req_params_from_openai(params)?;
    let mut tokens = req_input.tokens;
    let prompt = req_input.prompt;
    log::debug!("First input prompt {}", app_state.tok_env.tok_trie().tokens_dbg(&tokens));

    let eos_token = if is_chat {
        app_state.tok_eos_chat
    } else {
        app_state.tok_eos_completions
    };
    let eos_token = eos_token.unwrap_or_else(|| app_state.tok_env.eos_token());
    req_params.eos_token_id = eos_token;

    let client_req_id = app_state
        .next_client_req_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let client_req_id = ClientReqId::new(client_req_id as u64);
    let cmpl_id = format!("{}-{}", if is_run { "run" } else { "cmpl" }, Uuid::new_v4());

    let llg = if let Some(grm) = llg_grammar(params)? {
        let parser = app_state
            .parser_factory
            .create_parser_ext(grm, params.llg_log_level.to_log_level())?;

        let mut llg = Constraint::new(parser);

        if params.llg_log_level.has_json() {
            llg.log_json_progress = true;
        }

        // temperature handled by logits processing - this has to be 1.0
        // to avoid double-application of temperature
        llg.temperature = req_params.temperature;
        req_params.temperature = 1.0;

        req_params.use_logits_post_processor = true;

        // If we do that, we need to make sure we return the tokens forced
        // by the grammar to the user. Currently we don't have infra for that,
        // so instead we just start the parser without the prompt.
        //
        // if is_chat {
        //     tokens.extend_from_slice(&llg.process_prompt(vec![]));
        // } else {
        //     tokens = llg.process_prompt(tokens);
        // }
        llg.start_without_prompt();

        let mut r = vec![Box::new(llg)];
        while r.len() < req_params.num_return_sequences as usize {
            r.push(Box::new(r[0].deep_clone()));
        }
        r
    } else {
        if req_params.temperature < 0.00001 {
            req_params.temperature = 0.0;
            req_params.top_k = 1;
        }
        vec![]
    };

    if let Some(bos) = app_state.tok_bos {
        if tokens.is_empty() || tokens[0] != bos {
            tokens.insert(0, bos);
        }
    } else if tokens.is_empty() {
        // insert a space if all else fails
        tokens = app_state.tokenize_with_bos(" ");
    }

    let n_forks = req_params.num_return_sequences;

    // For our first attempt, only load LoRA weights if load_lora_weights is set to "always"
    let load_lora_weights = match params.load_lora_weights {
        LoadLoraWeightsOption::Auto => false,
        LoadLoraWeightsOption::Always => true,
        LoadLoraWeightsOption::Never => false,
    };
    let mut req_init = build_request_init(
        tokens.clone(),
        req_params.clone(),
        client_req_id,
        is_run,
        &params,
        app_state,
        load_lora_weights,
    )?;
    let prompt_tokens = req_init.tokens.len();

    // TODO draft executor call
    if AsyncExecutor::lock().has_draft_model() {
        // TODO override  n draft tokens to execute
        let start_len = req_init.tokens.len();
        let n_gen_tokens = req_init.params.max_new_tokens as usize;
        let total_len = start_len + n_gen_tokens;
        let n_draft_tokens = AsyncExecutor::lock().n_draft_tokens(); // TODO how long to do this
        let mut target_req_info: Option<ReqInfo> = None;  // need as option due to loop
        let mut draft_req_info: Option<ReqInfo> = None;  // need as option due to loop
        let mut gen_bytes: Vec<u8> = Vec::new();
        while req_init.tokens.len() < total_len {
            // handle case where there are less than n_draft_tokens + 1 needed
            let n_gen_tokens_cur_iter: usize = min(n_draft_tokens as usize + 1, total_len - req_init.tokens.len());
            let n_draft_tokens_cur_iter = min(n_gen_tokens_cur_iter - 1, n_draft_tokens as usize);
            let mut log_probs: Vec<TopTokenLogProb>;
            let mut req_id: ReqId;
            let mut recv;
            let mut draft_tokens = None;
            if n_draft_tokens_cur_iter > 0 { // if only need 1 token just skip to target
                log::debug!(
                    "Generate {} draft tokens",
                    n_draft_tokens_cur_iter
                );
                req_init.params.max_new_tokens = n_draft_tokens_cur_iter as u32;  // TODO set min?
                req_init.params.streaming = false; // Set to false for draft so that we can grab logits in one go.
                let (req_id, recv) = AsyncExecutor::lock().add_draft_request(
                    &req_init,
                    req_input.prompt_params.clone(),  // TODO needed here?
                    llg.clone(),
                )?;

                draft_req_info = draft_req_info.or(Some(build_req_info(
                    req_id,
                    1,  // TODO how to handle num_return_sequences? keep as 1 for now
                    client_req_id,
                    cmpl_id.clone(),
                    &prompt,
                    prompt_tokens,
                    params,
                    &app_state.tok_env,
                    is_chat,
                    is_run,
                    recv,
                )?));

                // TODO proper error handling,
                // TODO this doesn't need return passed reqinfo
                let (mut req_info_temp, _, mut logits_tensor) = gather_response_chunks(draft_req_info.unwrap()).await?;
                let cur_draft_tokens = req_info_temp.tok_env.tokenize_bytes(&req_info_temp.forks[0].text);

                log::debug!(
                    "Draft output: {}",
                    req_info_temp.tok_env.tok_trie().tokens_dbg(&cur_draft_tokens)
                );

                // TODO make this an assertion ideally?
                if cur_draft_tokens.len() != n_draft_tokens_cur_iter {
                    log::warn!("expected {} draft tokens got {}", n_draft_tokens_cur_iter, cur_draft_tokens.len())
                }

                if logits_tensor
                    .as_ref()
                    .and_then(|t| t.size.first().copied())
                    .map(|first_dim| first_dim as usize != cur_draft_tokens.len())
                    .unwrap_or(false)
                {
                    log::warn!("Logits tensor size does not match num tokens, not using logits.");
                    logits_tensor = None;
                }

                req_init.draft_params = Some(DraftParams {
                    draft_tokens: cur_draft_tokens.clone(),
                    logits_tensor: logits_tensor,
                    num_tokens: cur_draft_tokens.len() as u32  // TODO init correctly>
                });

                draft_req_info = Some(req_info_temp);
                draft_tokens = Some(cur_draft_tokens);
            } else {
                req_init.draft_params = None; // clear from last time draft model was called
            }

            let n_max_target_tokens = n_draft_tokens_cur_iter as u32 + 1;
            req_init.params.max_new_tokens = n_max_target_tokens;
            req_init.params.streaming = true; // Set to true for target model so that we can stream the response.
            (req_id, recv) = AsyncExecutor::lock().add_request(
                &req_init,
                req_input.prompt_params.clone(),
                llg.clone()
            )?;

            target_req_info = target_req_info.or(Some(build_req_info(
                req_id,
                1, // TODO how to handle num_return_sequences? keep as 1 for now,
                client_req_id,
                cmpl_id.clone(),
                &prompt,
                prompt_tokens,
                params,
                &app_state.tok_env,
                is_chat,
                is_run,
                recv,
            )?));

            let (mut req_info_temp, _, _) = gather_response_chunks(target_req_info.unwrap()).await?;
            let mut target_tokens = req_info_temp.tok_env.tokenize_bytes(&req_info_temp.forks[0].text);

            if let Some(draft_tokens) = draft_tokens {
                // draft tokens: [a, b, c, d, e]
                // target tokens: [a, b, c, y]
                // expect: 60% draft token usage
                // sub 1 from target token length for token target model will always produce
                 let perc_draft_tokens_used = 1.0 - (((draft_tokens.len() as f32) - ((target_tokens.len() - 1) as f32)) / (draft_tokens.len() as f32));
                 log::debug!(
                    "Target model used {:.2}% of draft model tokens",
                    perc_draft_tokens_used * 100 as f32
                );
            }

            log::debug!(
                "Target model generated {}/{} tokens", target_tokens.len(), n_max_target_tokens
            );

            log::debug!(
                "Target output: {}",
                req_info_temp.tok_env.tok_trie().tokens_dbg(&target_tokens)
            );

            req_init.tokens.append(&mut target_tokens);
            gen_bytes.append(&mut req_info_temp.forks[0].text);
            // TODO double check gather_response_chunks return prompt tokens as well?
            req_init = build_request_init(
                req_init.tokens,
                req_params.clone(),
                client_req_id,
                is_run,
                &params,
                app_state,
                load_lora_weights,
            )?;
            log::debug!(
                "Req init output: {}",
                req_info_temp.tok_env.tok_trie().tokens_dbg(&req_init.tokens)
            );
            target_req_info = Some(req_info_temp);

            // account for other stop conditions
            if let Some(stop_reason) = &target_req_info.as_ref().unwrap().forks[0].stop_reason {
                if stop_reason == &FinishReason::EosToken || stop_reason == &FinishReason::StopWords {
                    break;
                }
            }
        }

        // TODO if req_info has gotten all info should skip inner gather loop?
        // TODO need to redo n_completion_tokens for final req_info
        completions_stream_or_not(false, target_req_info.expect("no tokens generated"), Some(gen_bytes)).await
    } else {
        let (req_id, recv) = AsyncExecutor::lock().add_request(
            &req_init,
            req_input.prompt_params.clone(),
            llg.clone(),
        )?;

        let info = build_req_info(
            req_id,
            n_forks,
            client_req_id,
            cmpl_id.clone(),
            &prompt,
            prompt_tokens,
            params,
            &app_state.tok_env,
            is_chat,
            is_run,
            recv,
        )?;

        match completions_stream_or_not(params.stream, info, None).await {
            Ok(r) => Ok(r),
            Err(err) => {
                if is_lora_cache_miss_error(&err) {
                    log::warn!("LoRA cache miss: {}", err);

                    // If necessary, retry with LoRA weights set
                    if params.load_lora_weights == LoadLoraWeightsOption::Auto {
                        log::info!("Retrying with LoRA weights set");
                        let req_init = build_request_init(
                            tokens.clone(),
                            req_params.clone(),
                            client_req_id,
                            is_run,
                            &params,
                            app_state,
                            true,
                        )?;
                        let prompt_tokens = req_init.tokens.len();

                        let (req_id, recv) = AsyncExecutor::lock().add_request(
                            &req_init,
                            req_input.prompt_params.clone(),
                            llg.clone(),
                        )?;

                        let info = build_req_info(
                            req_id,
                            n_forks,
                            client_req_id,
                            cmpl_id.clone(),
                            &prompt,
                            prompt_tokens,
                            params,
                            &app_state.tok_env,
                            is_chat,
                            is_run,
                            recv,
                        )?;

                        completions_stream_or_not(params.stream, info, None).await
                    } else {
                        Ok(AppError::from(anyhow!(
                                "LoRA model {:?} was not in cache and load_lora_weights is set to never: {}", params.lora_model, err
                            )).into_response())
                    }
                } else {
                    Ok(err.into_response())
                }
            }
        }
    }
}

// #[axum::debug_handler]
pub async fn route_completions(
    _headers: HeaderMap,
    State(app_state): State<Arc<AppState>>,
    mut request: Json<CompletionCreateParams>,
) -> Result<Response, AppError> {
    log::debug!("request: {:?}", request);

    request.params.logprobs = request.logprobs;

    if let Err(e) = validate_compl(&request) {
        return Err(e.into());
    }

    let req_input = RequestInput::from_text(&app_state, &request.prompt[0]);

    mk_req_info(&app_state, req_input, &request.params, false, false)
        .await
        .map_err(|e| {
            log::warn!("completion error: {}", e);
            e
        })
}

pub async fn route_chat_completions(
    _headers: HeaderMap,
    State(app_state): State<Arc<AppState>>,
    mut request: Json<ChatCompletionCreateParams>,
) -> Result<Response, AppError> {
    log::debug!("chat request: {:?}", request);

    if request.logprobs {
        request.params.logprobs = Some(request.top_logprobs.unwrap_or(1));
    }

    if let Err(e) = validate_chat(&request) {
        return Err(e.into());
    }

    if request.tools.len() > 0 {
        let schema = tools_to_schema(&request.tools);
        log::debug!("tools schema: {}", serde_json::to_string_pretty(&schema)?);

        let lark_grm_templ = match &request.tool_choice {
            ToolChoice::Simple(ToolChoiceOption::None) => r"start: /(.|\n)*/",
            ToolChoice::Simple(ToolChoiceOption::Auto) => r"start: /[^{](.|\n)*/ | {json_start} @json_schema",
            ToolChoice::Simple(ToolChoiceOption::Required) | ToolChoice::Advanced(_) => {
                r"start: {json_start} @json_schema"
            }
        };

        let json_start = app_state.json_start_token_name.as_ref().map_or("", |s| s);
        let lark_grm = lark_grm_templ.replace("{json_start}", json_start);

        let mut grammar = TopLevelGrammar::from_lark(lark_grm);
        grammar
            .grammars
            .push(GrammarWithLexer::from_json_schema(schema));

        log::debug!("tools grammar: {}", serde_json::to_string(&grammar)?);
        request.params.response_format = Some(ResponseFormat::Llguidance { grammar });
    }

    let chat_params = if request.include_json_schema_in_prompt.unwrap_or(true) {
        let json_schema = match &request.params.response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => json_schema.schema.as_ref(),
            _ => None,
        };
        ChatParams {
            messages: &request.messages,
            tools: &request.tools,
            json_schema,
        }
    } else {
        // skip schema in prompt
        ChatParams {
            messages: &request.messages,
            tools: &vec![],
            json_schema: None,
        }
    };

    let req_input = if app_state.py_state.enabled {
        let req_params = req_params_from_openai(&request.params)?;
        app_state
            .py_state
            .run_input_processor(chat_params, &req_params)
            .map_err(|e| {
                log::warn!("chat py error: {}", e);
                e
            })?
    } else {
        let text = app_state.chat_builder.build(chat_params)?;
        RequestInput::from_text(&app_state, &text)
    };

    mk_req_info(&app_state, req_input, &request.params, true, false)
        .await
        .map_err(|e| {
            log::warn!("chat error: {}", e);
            e
        })
}

fn json_to_llg(schema: Value) -> Result<TopLevelGrammar> {
    Ok(TopLevelGrammar::from_json_schema(schema))
}

fn valid_utf8_len(data: &Vec<u8>) -> usize {
    if data.is_empty() {
        return 0;
    }

    // Find where the last valid UTF-8 sequence starts by scanning the final bytes
    let mut i = data.len() - 1;

    // Check if we have a continuation byte (0b10xxxxxx)
    while i > 0 && (data[i] & 0b1100_0000 == 0b1000_0000) {
        i -= 1;
    }

    // Check how many bytes the starting byte indicates for the UTF-8 sequence
    let first_byte = data[i];
    let expected_len = if first_byte & 0b1000_0000 == 0 {
        1 // Single-byte character (ASCII)
    } else if first_byte & 0b1110_0000 == 0b1100_0000 {
        2 // Two-byte character
    } else if first_byte & 0b1111_0000 == 0b1110_0000 {
        3 // Three-byte character
    } else if first_byte & 0b1111_1000 == 0b1111_0000 {
        4 // Four-byte character
    } else {
        1 // Invalid UTF-8, truncate it
    };

    // If there aren't enough bytes left for a valid character, truncate
    if i + expected_len <= data.len() {
        i + expected_len
    } else {
        i
    }
}

fn take_final_logs(llg: &mut Constraint) -> Result<String> {
    log::info!("llg-max-step: {:?}", llg.parser.max_step_stats());

    if !llg.log_json_progress && llg.parser.logger.buffer_level() == 0 {
        return Ok(String::new());
    }

    let eos = llg.tok_trie().eos_token();
    let res = llg.commit_token(Some(eos))?;
    if !res.stop {
        let _res = llg.compute_mask()?;
    }
    Ok(llg.flush_logs())
}

struct ReqCancelToken {
    req_id: Option<ReqId>,
}

impl ReqCancelToken {
    pub fn disarm(&mut self) {
        self.req_id = None;
    }
}

impl Drop for ReqCancelToken {
    fn drop(&mut self) {
        if let Some(req_id) = self.req_id {
            log::warn!("dropping ReqCancelToken: {}", req_id);
            let _ = AsyncExecutor::lock().cancel_request(req_id);
        } else {
            log::debug!("not dropping ReqCancelToken");
        }
    }
}

impl ReqInfo {
    fn cancel_token(&self) -> ReqCancelToken {
        ReqCancelToken {
            req_id: Some(self.req_id),
        }
    }

    fn all_forks_stopped(&self) -> bool {
        self.forks.iter().all(|f| f.stop_reason.is_some())
    }

    /// This returns the added text in the response, limited to valid UTF8 length.
    /// The fork.text is updated with all new bytes, not only valid UTF8.
    fn update_text(&mut self, result: &mut StepResults, incremental: bool) -> RunForkResponse {
        let idx = result.response.sequence_idx as usize;

        let fork = &mut self.forks[idx];

        if let Some(l) = result.take_logs() {
            fork.logs.push_str(&l);
        }

        if let Some(mut llg) = result.final_llg.take() {
            match take_final_logs(llg.as_mut()) {
                Ok(logs) => fork.logs.push_str(&logs),
                Err(e) => {
                    if result.response.error.is_none() {
                        result.response.error = Some(format!("{e}"));
                    }
                    let msg = format!("error taking final logs: {e}");
                    log::debug!("{}", msg);
                    fork.logs.push_str("\nWarning: ");
                    fork.logs.push_str(&msg);
                }
            }
        }

        let logprobs = result.response.logprobs.as_ref().map(|lp| {
            let trie = self.tok_env.tok_trie();
            LogProbs {
                content: lp.iter().map(|v| TopTokenLogProb::new(trie, v)).collect(),
            }
        });

        let mut text = String::new();

        if fork.stop_reason.is_none() {
            let response = &result.response;
            fork.stop_reason = response.finish_reason.clone();

            let mut bytes = std::mem::replace(&mut fork.pending_bytes, Vec::new());
            let new_bytes = self.tok_env.tok_trie().decode(&response.tokens);
            bytes.extend_from_slice(&new_bytes);

            fork.text.extend_from_slice(&new_bytes);

            text = 'outer: {
                if self.stop_words.len() > 0 {
                    let max_suffix = new_bytes.len()
                        + self.stop_words.iter().map(|s| s.len()).max().unwrap_or(0);
                    let start_pos = fork.text.len().saturating_sub(max_suffix);
                    for stop in &self.stop_words {
                        if let Some(stop_off) = fork.text[start_pos..]
                            .windows(stop.len())
                            .position(|w| w == stop)
                        {
                            let pos = start_pos + stop_off;
                            let drop_len = fork.text.len() - pos;
                            fork.text.truncate(pos);
                            fork.stop_reason = Some(trtllm_rs::FinishReason::StopWords);
                            if drop_len >= bytes.len() {
                                // No more bytes to emitted from last chunk
                                break 'outer String::new();
                            } else {
                                bytes.truncate(bytes.len() - drop_len);
                                break 'outer String::from_utf8_lossy(&bytes).to_string();
                            }
                        }
                    }
                }

                let pref_len = valid_utf8_len(&bytes);
                fork.pending_bytes = bytes.split_off(pref_len);
                String::from_utf8_lossy(&bytes).to_string()
            };
        }

        let logs = if incremental {
            std::mem::take(&mut fork.logs)
        } else {
            String::new()
        };

        RunForkResponse {
            index: idx,
            finish_reason: fork.stop_reason.clone().map(map_finish_reason),
            text,
            error: result.response.error.clone().unwrap_or_default(),
            logs,
            storage: Vec::new(),
            micros: 0,
            logprobs,
        }
    }

    fn run_chunk(&mut self, mut result: StepResults) -> RunResponse {
        let resp = self.update_text(&mut result, true);
        RunResponse {
            object: "run",
            forks: vec![resp],
            usage: RunUsageResponse {
                sampled_tokens: self.usage.completion_tokens,
                ff_tokens: self.usage.total_tokens,
                cost: 2 * self.usage.completion_tokens + self.usage.total_tokens,
            },
        }
    }

    fn chat_completion(&mut self, mut result: StepResults) -> ChatCompletionChunk {
        let resp = self.update_text(&mut result, true);

        ChatCompletionChunk {
            id: self.cmpl_id.clone(),
            object: "text_completion".to_string(),
            created: self.created,
            model: self.model_name.clone(),
            system_fingerprint: None,
            choices: vec![ChatCompletionChunkChoice {
                index: resp.index,
                delta: ChatCompletionChunkDelta {
                    role: Some(Role::Assistant),
                    content: Some(resp.text),
                },
                finish_reason: resp.finish_reason,
                llg_logs: if resp.logs.is_empty() {
                    None
                } else {
                    Some(resp.logs)
                },
                logprobs: resp.logprobs,
            }],
            usage: self.usage.clone(),
        }
    }

    fn initial_run(&mut self) -> Event {
        Event::default()
            .json_data(InitialRunResponse {
                id: self.cmpl_id.clone(),
                object: "initial-run",
                created: self.created,
                model: self.model_name.clone(),
            })
            .unwrap()
    }

    fn resp_to_event(&mut self, result: StepResults) -> Event {
        let response = &result.response;
        if !self.is_run {
            if let Some(err) = &response.error {
                log::error!("received error message: {}", err);
                return Event::default()
                    // .event("error")
                    .json_data(json!({
                        "error": {
                            "status_code": 500,
                            "message": err
                        }
                    }))
                    .unwrap();
            }
        }

        log::trace!("infer response: {:?}", response);
        self.usage.completion_tokens += response.tokens.len();
        self.usage.total_tokens += response.tokens.len();

        if self.is_run {
            Event::default().json_data(self.run_chunk(result)).unwrap()
        } else if self.is_chat {
            Event::default()
                .json_data(self.chat_completion(result))
                .unwrap()
        } else {
            Event::default()
                .json_data(Completion::of_chat_completion_chunk(
                    self.chat_completion(result),
                ))
                .unwrap()
        }
    }
}

async fn completions_stream(
    mut client: ReqInfo,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, AppError> {
    let mut token = client.cancel_token();
    let result0 = client
        .recv
        .recv()
        .await
        .ok_or_else(|| anyhow!("no response"))?;
    if let Some(err) = result0.response.error {
        // return as HTTP error, not error event
        return Err(anyhow!("{}", err).into());
    }

    let response_stream = try_stream! {
        let mut token = client.cancel_token();

        if client.is_run {
            yield client.initial_run();
        }

        yield client.resp_to_event(result0);

        while let Some(result) = client.recv.recv().await {
            let is_error = result.response.error.is_some();
            yield client.resp_to_event(result);
            if is_error {
                return;
            }
            if client.all_forks_stopped() {
                let _ = AsyncExecutor::lock().cancel_request(client.req_id);
            }
        }

        yield Event::default().data("[DONE]");

        token.disarm();
    };

    token.disarm();
    Ok(Sse::new(response_stream))
}

async fn gather_response_chunks(mut client: ReqInfo) -> Result<(ReqInfo, Vec<TopTokenLogProb>, Option<Tensor>), Error> {
    let mut logprobs = vec![];
    let mut logits = vec![];
    while let Some(mut result) = client.recv.recv().await {
        log::trace!("infer response: {:?}", result.response);
        let response = &result.response;
        if let Some(err) = &response.error {
            let err = anyhow::anyhow!("{}", err);
            log::error!("received error message (rest): {}", err);
            let _ = AsyncExecutor::lock().cancel_request(client.req_id);
            return Err(err.into());
        } else {
            client.usage.completion_tokens += response.tokens.len();
            client.usage.total_tokens += response.tokens.len();
            let response = result.response.clone(); // Clone the response to avoid borrowing conflicts
            let r = client.update_text(&mut result, false);
            if let Some(mut lp) = r.logprobs {
                logprobs.append(&mut lp.content);
            }
            if let Some(ref l) = response.generation_logits {
                if !l.size.is_empty() && !l.data.is_empty() {
                    logits.push(Tensor::from(l.clone())); // TODO: Revisit if we ensure only one logit will ever be returned.
                } else {
                    log::debug!("no logits detected")
                }
            } else {
                log::debug!("no logits detected")
            }
        }

        if client.all_forks_stopped() {
            let _ = AsyncExecutor::lock().cancel_request(client.req_id);
            break;
        }
    }

    for (i, logit) in logits.iter().enumerate() {
        log::debug!("Logit {} shape: {:?}", i, logit.size.iter().map(|&x| x as u32).collect::<Vec<u32>>());
    }

    let logits_tensor = logits.get(0).cloned();

    Ok((client, logprobs, logits_tensor))
}

async fn completions(mut client: ReqInfo, final_content: Option<Vec<u8>>) -> Result<Json<Value>, AppError> {
    let mut token = client.cancel_token();
    let mut logprobs = vec![];

    // TODO this should skip this if gather_response_chunks was called already
    if client.all_forks_stopped() {
        (client, logprobs, _) = gather_response_chunks(client).await?;
    }
    // while let Some(mut result) = client.recv.recv().await {
    //     log::trace!("infer response: {:?}", result.response);
    //     let response = &result.response;
    //     if let Some(err) = &response.error {
    //         let err = anyhow::anyhow!("{}", err);
    //         log::error!("received error message (rest): {}", err);
    //         let _ = AsyncExecutor::lock().cancel_request(client.req_id);
    //         return Err(err.into());
    //     } else {
    //         client.usage.completion_tokens += response.tokens.len();
    //         client.usage.total_tokens += response.tokens.len();
    //         let r = client.update_text(&mut result, false);
    //         if let Some(mut lp) = r.logprobs {
    //             logprobs.append(&mut lp.content);
    //         }
    //     }

    //     if client.all_forks_stopped() {
    //         let _ = AsyncExecutor::lock().cancel_request(client.req_id);
    //         break;
    //     }
    // }

    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let id = format!("cmpl-{}", Uuid::new_v4());

    let chat_compl = ChatCompletion {
        id,
        object: "text_completion".to_string(),
        created,
        model: client.model_name,
        system_fingerprint: None,
        expanded_prompt: if client.return_expanded_prompt {
            Some(client.prompt)
        } else {
            None
        },
        choices: client
            .forks
            .into_iter()
            .enumerate()
            .map(|(index, fork)| ChatCompletionChoice {
                index,
                message: ChatCompletionMessage {
                    role: Role::Assistant,
                    content: Some(String::from_utf8_lossy(&final_content.clone().unwrap_or(fork.text)).to_string()),
                },
                finish_reason: fork.stop_reason.map(map_finish_reason),
                llg_logs: if fork.logs.is_empty() {
                    None
                } else {
                    Some(fork.logs)
                },
                logprobs: if logprobs.is_empty() {  // TODO need to pass logprobs if gather_response_chunks called earlier
                    None
                } else {
                    Some(LogProbs {
                        content: std::mem::take(&mut logprobs),
                    })
                },
            })
            .collect(),
        usage: client.usage,
    };

    let inner = if client.is_chat {
        serde_json::to_value(chat_compl)?
    } else {
        serde_json::to_value(Completion::of_chat_completion(chat_compl))?
    };

    token.disarm();

    Ok(Json(inner))
}

fn validate_run(req: &RunRequest, common: &CommonCreateParams) -> Result<()> {
    ensure!(
        req.prompt.is_none() || req.messages.is_none(),
        "prompt and messages are mutually exclusive"
    );
    ensure!(
        req.controller == "llguidance",
        "controller must be 'llguidance'"
    );

    let _ = req_params_from_openai(common)?;

    Ok(())
}

pub async fn route_llguidance(
    _headers: HeaderMap,
    State(app_state): State<Arc<AppState>>,
    request: Json<RunRequest>,
) -> Result<Response, AppError> {
    log::debug!("run request: {:?}", request);

    let mut common: CommonCreateParams = serde_json::from_value(json!({
        "model": "model"
    }))?;

    if let Some(v) = request.temperature {
        common.temperature = v;
    }
    if let Some(v) = request.top_p {
        common.top_p = v;
    }
    common.max_tokens = request.max_tokens;

    common.stream = true;
    common.llg_log_level = LlgLogLevel::Verbose;
    common.response_format = Some(ResponseFormat::Llguidance {
        grammar: request.controller_arg.grammar.clone(),
    });

    if let Err(e) = validate_run(&request, &common) {
        return Err(e.into());
    }

    let mut is_chat = false;

    let chat_history = if let Some(messages) = &request.messages {
        is_chat = true;
        app_state.chat_builder.build(ChatParams {
            messages,
            tools: &vec![],
            json_schema: None,
        })?
    } else {
        request.prompt.clone().unwrap_or(String::new())
    };
    let req_input = RequestInput::from_text(&app_state, &chat_history);

    mk_req_info(&app_state, req_input, &common, is_chat, true)
        .await
        .map_err(|e| {
            log::warn!("llg error: {}", e);
            e
        })
}

fn safetensors_dtype_to_tlc(dtype: Dtype) -> TlcDataType {
    match dtype {
        Dtype::BOOL => TlcDataType::TLC_DT_BOOL,
        Dtype::U8 => TlcDataType::TLC_DT_U8,
        Dtype::I8 => TlcDataType::TLC_DT_I8,
        Dtype::F8_E4M3 => TlcDataType::TLC_DT_F8,
        Dtype::F8_E5M2 => panic!("F8_E5M2 not supported"), // we support the other one
        Dtype::F16 => TlcDataType::TLC_DT_F16,
        Dtype::BF16 => TlcDataType::TLC_DT_BF16,
        Dtype::I32 => TlcDataType::TLC_DT_I32,
        Dtype::F32 => TlcDataType::TLC_DT_F32,
        Dtype::I64 => TlcDataType::TLC_DT_I64,

        Dtype::U64 | Dtype::U32 | Dtype::F64 | Dtype::I16 | Dtype::U16 | _ => {
            panic!("unsupported dtype: {:?}", dtype)
        }
    }
}
