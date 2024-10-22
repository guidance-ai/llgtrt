//! https://platform.openai.com/docs/api-reference/completions/create
use anyhow::{bail, ensure, Result};
use async_stream::try_stream;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_core::Stream;
use llguidance_parser::api::TopLevelGrammar;
use llguidance_parser::{Constraint, JsonCompileOptions};
use serde_json::{json, Value};
use std::fmt::Display;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use toktrie::TokEnv;
use trtllm_rs::{ClientReqId, ReqId, RequestInit, RequestParams, TokenId};
use uuid::Uuid;

use crate::async_exec::{map_finish_reason, AsyncExecutor, StepResults};
use crate::constraint_mgr::ConstraintInit;
use crate::error::AppError;
use crate::routes::api_ext::LlgLogLevel;
use crate::routes::openai::ResponseFormat;
use crate::state::AppState;

use super::api_ext::{
    InitialRunResponse, RunForkResponse, RunRequest, RunResponse, RunUsageResponse,
};
use super::openai::{
    ChatCompletion, ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
    ChatCompletionChunkDelta, ChatCompletionCreateParams, ChatCompletionMessage,
    CommonCreateParams, Completion, CompletionCreateParams, LogProbs, Role, TopTokenLogProb, Usage,
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
        Some(ResponseFormat::JsonSchema { name: Some(n), .. }) => {
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

    if let Some(seed) = params.seed {
        r.seed = seed;
    } else {
        r.seed = rand::random();
    }

    Ok(r)
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
    Ok(())
}

fn llg_grammar(params: &CommonCreateParams) -> Result<Option<TopLevelGrammar>> {
    let opts = JsonCompileOptions { compact: false };
    let grm = match &params.response_format {
        Some(ResponseFormat::Llguidance { grammar }) => grammar.clone(),
        Some(ResponseFormat::JsonObject)
        | Some(ResponseFormat::JsonSchema { strict: false, .. }) => {
            opts.json_to_llg(&json!({ "type": "object" }))?
        }
        Some(ResponseFormat::JsonSchema {
            schema: None,
            strict: true,
            ..
        }) => {
            bail!("missing schema in strict mode")
        }
        Some(ResponseFormat::JsonSchema {
            schema: Some(schema),
            strict: true,
            ..
        }) => opts.json_to_llg(schema)?,
        _ => return Ok(None),
    };
    Ok(Some(grm))
}

async fn mk_req_info(
    app_state: &AppState,
    mut tokens: Vec<TokenId>,
    params: &CommonCreateParams,
    is_chat: bool,
    is_run: bool,
) -> Result<Response, AppError> {
    let mut req_params = req_params_from_openai(params)?;

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
        // println!("grammar: {}", serde_json::to_string(&grm).unwrap());
        let mut llg = app_state.constraint_mgr.new_constraint(ConstraintInit {
            grammar: grm,
            is_chat,
            log_level: params.llg_log_level,
        })?;

        // temperature handled by logits processing - this has to be 1.0
        // to avoid double-application of temperature
        llg.temperature = req_params.temperature;
        req_params.temperature = 1.0;

        // Test argmax
        // llg.temperature = 1.0;
        // req_params.temperature = 0.0;
        // req_params.top_k = 1;

        req_params.use_logits_post_processor = true;

        if is_chat {
            tokens.extend_from_slice(&llg.process_prompt(vec![]));
        } else {
            tokens = llg.process_prompt(tokens);
        }

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
        if tokens.len() == 0 || tokens[0] != bos {
            tokens.insert(0, bos);
        }
    } else if tokens.len() == 0 {
        // insert a space if all else fails
        tokens = app_state.tokenize_with_bos(" ");
    }

    let n_forks = req_params.num_return_sequences;

    let req_init = RequestInit {
        tokens,
        params: req_params,
        client_req_id,
        is_run,
    };
    let prompt_tokens = req_init.tokens.len();

    let (req_id, recv) = AsyncExecutor::lock().add_request(req_init, llg)?;

    let info = ReqInfo {
        req_id,
        client_req_id,
        is_chat,
        is_run,
        cmpl_id,
        model_name: params.model.clone(),
        created: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        recv,
        tok_env: app_state.tok_env.clone(),
        forks: vec![ForkInfo::default(); n_forks as usize],
        stop_words: params
            .stop
            .as_ref()
            .map(|stops| stops.iter().map(|s| s.as_bytes().to_vec()).collect())
            .unwrap_or_default(),
        usage: Usage {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        },
    };

    if params.stream {
        Ok(completions_stream(info).await.into_response())
    } else {
        Ok(completions(info).await.into_response())
    }
}

pub async fn route_completions(
    _headers: HeaderMap,
    State(app_state): State<Arc<AppState>>,
    mut request: Json<CompletionCreateParams>,
) -> Result<Response, AppError> {
    log::info!("request: {:?}", request);

    request.params.logprobs = request.logprobs;

    if let Err(e) = validate_compl(&request) {
        return Err(e.into());
    }

    let tokens = app_state.tokenize_with_bos(&request.prompt[0]);
    mk_req_info(&app_state, tokens, &request.params, false, false).await
}

pub async fn route_chat_completions(
    _headers: HeaderMap,
    State(app_state): State<Arc<AppState>>,
    mut request: Json<ChatCompletionCreateParams>,
) -> Result<Response, AppError> {
    log::info!("chat request: {:?}", request);

    if request.logprobs {
        request.params.logprobs = Some(request.top_logprobs.unwrap_or(1));
    }

    if let Err(e) = validate_chat(&request) {
        return Err(e.into());
    }

    let chat_history = app_state.chat_builder.build(&request.messages)?;
    log::debug!("chat history after formatting:\n{:?}", chat_history);

    let tokens = app_state.tokenize_with_bos(&chat_history);
    log::debug!("{}", app_state.tok_env.tok_trie().tokens_dbg(&tokens));

    mk_req_info(&app_state, tokens, &request.params, true, false).await
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

impl ReqInfo {
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
                // Corresponds to https://github.com/openai/openai-python/blob/17ac6779958b2b74999c634c4ea4c7b74906027a/src/openai/_streaming.py#L113
                return Event::default()
                    .event("error")
                    .json_data(json!({
                        "error": {
                            "status_code": 500,
                            "message": err
                        }
                    }))
                    .unwrap();
            }
        }

        log::debug!("infer response: {:?}", response);
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
    let response_stream = try_stream! {
        if client.is_run {
            yield client.initial_run();
        }

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
    };

    Ok(Sse::new(response_stream).keep_alive(KeepAlive::default()))
}

async fn completions(mut client: ReqInfo) -> Result<Json<Value>, AppError> {
    let mut logprobs = vec![];
    while let Some(mut result) = client.recv.recv().await {
        log::debug!("infer response: {:?}", result.response);
        let response = &result.response;
        if let Some(err) = &response.error {
            let err = anyhow::anyhow!("{}", err);
            log::error!("received error message (rest): {}", err);
            let _ = AsyncExecutor::lock().cancel_request(client.req_id);
            return Err(err.into());
        } else {
            client.usage.completion_tokens += response.tokens.len();
            client.usage.total_tokens += response.tokens.len();
            let r = client.update_text(&mut result, false);
            if let Some(mut lp) = r.logprobs {
                logprobs.append(&mut lp.content);
            }
        }

        if client.all_forks_stopped() {
            let _ = AsyncExecutor::lock().cancel_request(client.req_id);
            break;
        }
    }

    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let id = format!("cmpl-{}", Uuid::new_v4());

    let chat_compl = ChatCompletion {
        id,
        object: "text_completion".to_string(),
        created,
        model: client.model_name,
        system_fingerprint: None,
        choices: client
            .forks
            .into_iter()
            .enumerate()
            .map(|(index, fork)| ChatCompletionChoice {
                index,
                message: ChatCompletionMessage {
                    role: Role::Assistant,
                    content: Some(String::from_utf8_lossy(&fork.text).to_string()),
                },
                finish_reason: fork.stop_reason.map(map_finish_reason),
                llg_logs: if fork.logs.is_empty() {
                    None
                } else {
                    Some(fork.logs)
                },
                logprobs: if logprobs.is_empty() {
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
    log::info!("run request: {:?}", request);

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
        app_state.chat_builder.build(messages)?
    } else {
        request.prompt.clone().unwrap_or(String::new())
    };
    log::debug!("chat history after formatting:\n{:?}", chat_history);

    let tokens = app_state.tokenize_with_bos(&chat_history);
    log::debug!("{}", app_state.tok_env.tok_trie().tokens_dbg(&tokens));

    mk_req_info(&app_state, tokens, &common, is_chat, true).await
}
