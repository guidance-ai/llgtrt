use anyhow::{ensure, Result};
use llguidance::Constraint;
use rayon::prelude::*;
use std::{
    any::Any,
    collections::HashMap,
    fmt::Display,
    panic::{self, AssertUnwindSafe},
    ptr,
    sync::{Arc, Mutex, MutexGuard},
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use toktrie::{SimpleVob, TokEnv};
use trtllm_rs::{
    ClientReqId, Executor, ExecutorInit, MaskAllocator, ReqId, RequestInit, ResponseChunk, TlcLogitsEntry
};

use crate::{
    chat::ChatBuilder,
    config::{CliConfig, LlgTrtConfig},
    py::PyPromptParams,
    routes::openai::FinishReason,
    tokenizer::setup_tokenizer,
};

pub struct StepResults {
    pub response: ResponseChunk,
    pub logs: String,
    pub final_llg: Option<Box<Constraint>>,
}

impl StepResults {
    pub fn take_logs(&mut self) -> Option<String> {
        if self.logs.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.logs))
    }
}

pub fn map_finish_reason(fr: trtllm_rs::FinishReason) -> FinishReason {
    match fr {
        trtllm_rs::FinishReason::EosToken | trtllm_rs::FinishReason::StopWords => {
            FinishReason::Stop
        }
        trtllm_rs::FinishReason::Length => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

struct ConstraintInfo {
    req_id: ReqId,
}

struct ReqData {
    req_id: ReqId,
    client_req_id: ClientReqId,
    tx: UnboundedSender<StepResults>,
    logs: String,
    llgs: Vec<Option<Box<Constraint>>>,
    // trtllm will create new req_id when n>1
    // it seems to create them one by one
    // this array keeps track of assignment of req_id to llg state
    llg_infos: Vec<ConstraintInfo>,
    min_p: f32,
    prompt_len: usize,
    is_run: bool,
    // this needs to be here, so the tensor memory passed to trtllm is not dropped
    _prompt_params: Option<Arc<PyPromptParams>>,
}

impl Display for ReqData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReqData({} {})", self.req_id, self.client_req_id)
    }
}

pub struct AsyncExecutor {
    executor: Executor,
    draft_executor: Option<Executor>,
    n_vocab: usize,
    max_batch_size: usize,
    req_to_client: HashMap<ReqId, ClientReqId>,
    req_data: HashMap<ClientReqId, ReqData>,
    n_draft_tokens: u32,  // TODO prob better location for this, esp if dynamically setting between calls
    draft_token_acc_rate: Option<f32>,
}

static mut GLOBAL_ALLOCATOR: *const MaskAllocator = ptr::null();
static mut GLOBAL_EXECUTOR: *const Mutex<AsyncExecutor> = ptr::null();

struct PendingSeq {
    entry_idx: usize,
    llg: Box<Constraint>,
    llg_idx: usize,
    prompt_len: usize,
    is_run: bool,
    entry: TlcLogitsEntry,
    min_p: f32,
    stop: bool,
    // setting this will stop the sequence with given error
    error: Option<String>,
}
unsafe impl Send for PendingSeq {}

fn copy_mask(src: &SimpleVob) -> *mut u32 {
    let dst = AsyncExecutor::mask_allocator().allocate();
    dst.copy_from_slice(&src.as_slice()[0..dst.len()]);
    dst.as_mut_ptr()
}

impl PendingSeq {
    fn step(&mut self) -> Result<()> {
        let tokens = unsafe { self.entry.tokens() };

        let llg = self.llg.as_mut();

        log::trace!("Tokens: {}", llg.tok_trie().tokens_dbg(tokens));

        let step_res = if tokens.len() > self.prompt_len {
            // if we're past the prompt, commit last token
            // and compute mask
            let tok = *tokens.last().unwrap();
            let r = llg.commit_token(Some(tok))?;

            assert!(r.backtrack == 0);
            if r.stop {
                self.stop = true;
                return Ok(());
            }

            assert!(r.ff_tokens.len() == 1);
            assert!(r.ff_tokens[0] == tok);
            llg.compute_mask()?
        } else {
            // if we're still in prompt
            if llg.step_result().sample_mask.is_none() {
                // first time, compute the mask
                llg.compute_mask()?
            } else {
                // if trtllm wants to call us multiple times for the prompt
                // (this happens due to chunked prefill), we re-use the first mask
                llg.step_result()
            }
        };

        if step_res.is_stop() {
            self.stop = true;
            return Ok(());
        }

        let mask = step_res.sample_mask.as_ref().expect("No mask");
        self.entry.out_mask_pointer = copy_mask(mask);
        self.entry.temperature = llg.temperature;
        self.entry.ln_min_p = if self.min_p > 0.0 {
            self.min_p.ln()
        } else {
            -f32::MAX
        };

        Ok(())
    }
}

impl PendingSeq {
    fn new(rd: &mut ReqData, entry: &TlcLogitsEntry, entry_idx: usize, llg_idx: usize) -> Self {
        let llg = std::mem::take(&mut rd.llgs[llg_idx]).unwrap();
        Self {
            entry_idx,
            llg,
            llg_idx,
            prompt_len: rd.prompt_len,
            entry: entry.clone(),
            stop: false,
            min_p: rd.min_p,
            error: None,
            is_run: rd.is_run,
        }
    }
}

fn mk_panic_error(info: &Box<dyn Any + Send>) -> String {
    let msg = match info.downcast_ref::<&'static str>() {
        Some(s) => *s,
        None => match info.downcast_ref::<String>() {
            Some(s) => &s[..],
            None => "non-string panic!()",
        },
    };

    format!("panic: {msg}")
}

extern "C" fn logits_processor(logits: *mut TlcLogitsEntry, num_logits: u32) {
    let entries = unsafe { std::slice::from_raw_parts_mut(logits, num_logits as usize) };

    let mut pending_seqs = vec![];

    // check out sequences
    {
        let mut exec = AsyncExecutor::lock();
        let mut pending_assignments = vec![];
        for (idx, entry) in entries.iter().enumerate() {
            if let Some(rd) = exec.req_data.get_mut(&entry.client_req_id()) {
                log::debug!(
                    "llg: {}: {} tokens ({} prompt tokens)",
                    entry.req_id(),
                    entry._num_tokens,
                    rd.prompt_len
                );
                if let Some(llg_idx) = rd
                    .llg_infos
                    .iter()
                    .position(|ci| ci.req_id == entry.req_id())
                {
                    pending_seqs.push(PendingSeq::new(rd, entry, idx, llg_idx));
                } else {
                    pending_assignments.push((entry._req_id, idx));
                }
            }
        }
        pending_assignments.sort_by_key(|(req_id, _)| *req_id);
        for (_, idx) in pending_assignments {
            let entry = &entries[idx];
            let rd = exec.req_data.get_mut(&entry.client_req_id()).unwrap();
            let llg_idx = rd.llg_infos.len();
            log::debug!(
                "assigning llg: {} = {}.{}",
                entry.req_id(),
                entry.client_req_id(),
                llg_idx
            );
            rd.llg_infos.push(ConstraintInfo {
                req_id: entry.req_id(),
            });
            assert!(llg_idx < rd.llgs.len());
            pending_seqs.push(PendingSeq::new(rd, entry, idx, llg_idx));
        }
    }

    // let fractions = AsyncExecutor::mask_allocator().mask_fractions(1);
    // log::info!("Mask fractions: {:?}", fractions);

    AsyncExecutor::mask_allocator().reset();

    let pending_seqs = pending_seqs
        .into_par_iter()
        .map(|mut ps| {
            let r = match panic::catch_unwind(AssertUnwindSafe(|| ps.step())) {
                Err(e) => Err(mk_panic_error(&e)),
                Ok(Err(e)) => Err(format!("{:?}", e)),
                Ok(Ok(ps)) => Ok(ps),
            };
            match r {
                Err(msg) => {
                    log::error!("llg error: {} {}", ps.entry, msg);
                    ps.stop = true;
                    ps.error = Some(msg);
                }
                Ok(()) => {
                    if ps.stop {
                        // /v1/run has its own handling of stop reasons
                        if !ps.is_run && !ps.llg.parser.stop_reason().is_ok() {
                            let msg = format!(
                                "llg stop reason: {}",
                                ps.llg.parser.stop_reason().to_string()
                            );
                            log::warn!("{}", msg);
                            ps.error = Some(msg);
                        } else {
                            assert!(ps.entry.out_mask_pointer.is_null());
                            let trie = ps.llg.tok_trie();
                            let only_eos = trie.singleton_token_set(trie.eos_token());
                            ps.entry.out_mask_pointer = copy_mask(&only_eos);
                        }
                    } else {
                        assert!(!ps.entry.out_mask_pointer.is_null());
                    }
                }
            }
            ps
        })
        .collect::<Vec<_>>();

    // check sequences back in
    {
        let mut exec = AsyncExecutor::lock();
        for ps in pending_seqs {
            let entry = &mut entries[ps.entry_idx];
            entry.out_mask_pointer = ps.entry.out_mask_pointer;
            entry.temperature = ps.entry.temperature;
            entry.ln_min_p = ps.entry.ln_min_p;
            let mut llg = ps.llg;
            if let Some(rd) = exec.req_data.get_mut(&entry.client_req_id()) {
                if rd.logs.is_empty() {
                    // no copy in common case
                    rd.logs = llg.flush_logs();
                } else {
                    rd.logs.push_str(&llg.flush_logs());
                }
                if let Some(err) = ps.error {
                    rd.logs.push_str(&format!("\nWarning: {}\n", err));
                    let _ = rd.tx.send(StepResults {
                        response: ResponseChunk {
                            req_id: entry.req_id(),
                            sequence_idx: ps.llg_idx as u32,
                            finish_reason: Some(trtllm_rs::FinishReason::Unknown),
                            error: Some(err),
                            is_req_final: true,
                            logprobs: None,
                            tokens: vec![],
                            generation_logits: None // TODO how fill this
                        },
                        logs: std::mem::take(&mut rd.logs),
                        final_llg: None,
                    });
                    let _ = exec.cancel_request(entry.req_id());
                } else {
                    rd.llgs[ps.llg_idx] = Some(llg);
                }
            } else {
                log::warn!("Sequence {} went missing while computing logit mask", entry);
            }
        }
    }
}

impl AsyncExecutor {
    pub fn set_global(executor: AsyncExecutor) {
        unsafe {
            if GLOBAL_EXECUTOR.is_null() {
                let mask_allocator = MaskAllocator::new(executor.n_vocab, executor.max_batch_size + 1);
                GLOBAL_ALLOCATOR = Box::leak(Box::new(mask_allocator));
                GLOBAL_EXECUTOR = Box::leak(Box::new(Mutex::new(executor)));
            } else {
                panic!("Global executor already set");
            }
        }
    }

    fn mask_allocator() -> &'static MaskAllocator {
        unsafe {
            if GLOBAL_ALLOCATOR.is_null() {
                panic!("Global allocator not initialized");
            }
            GLOBAL_ALLOCATOR.as_ref().unwrap()
        }
    }

    pub fn lock() -> MutexGuard<'static, AsyncExecutor> {
        unsafe {
            if GLOBAL_EXECUTOR.is_null() {
                panic!("Global executor not initialized");
            }
            GLOBAL_EXECUTOR.as_ref().unwrap().lock().unwrap()
        }
    }

    fn drop_request_data(&mut self, req_id: ReqId) {
        if let Some(client_req_id) = self.req_to_client.remove(&req_id) {
            let _ = self.req_data.remove(&client_req_id);
        }
    }

    pub fn cancel_request(&mut self, req_id: ReqId) -> Result<()> {
        self.drop_request_data(req_id);
        self.executor.cancel_request(req_id)
    }

    pub fn has_draft_model(&self) -> bool {
        self.draft_executor.is_some()
    }

    pub fn n_draft_tokens(&self) -> u32 {
        self.n_draft_tokens
    }

    pub fn draft_token_acc_rate(&self) -> Option<f32> {
        self.draft_token_acc_rate
    }

    pub fn new(
        cli_config: &CliConfig,
        config: &LlgTrtConfig,
        mut executor_init: ExecutorInit,
        draft_executor_init: Option<ExecutorInit>,
        n_draft_tokens: u32,
        draft_token_acc_rate: Option<f32>
    ) -> Result<(Self, TokEnv, ChatBuilder)> {
        // executor_init.logits_callback = Some(logits_processor);
        let mut max_batch_size = executor_init.trt_params.max_batch_size as usize;
        log::info!("new executor: max_batch_size={max_batch_size}");
        let (executor, mut responder) = Executor::new(executor_init)?;

        let (draft_executor, mut draft_responder) = if let Some(draft_executor_init) = draft_executor_init {
            // TODO don't set logit processor for draft model right now
            //draft_executor_init.logits_callback = Some(logits_processor);
            max_batch_size = draft_executor_init.trt_params.max_batch_size as usize;
            log::info!("new draft executor: max_batch_size={max_batch_size}");
            let (executor, responder) = Executor::new(draft_executor_init)?;
            (Some(executor), Some(responder))
        } else {
            (None, None)
        };

        // TODO need to do this for draft executor?
        // on non-0 ranks, this will just wait until the rank 0 exits and then exit the process
        executor.check_mpi();

        // only setup tokenizer on rank 0
        let (tok_env, chat_builder) = setup_tokenizer(cli_config, config)?;
        let trie = tok_env.tok_trie();
        let n_vocab = trie.vocab_size();

        let res = Self {
            executor,
            draft_executor,
            req_data: HashMap::new(),
            req_to_client: HashMap::new(),
            n_vocab,
            max_batch_size,
            n_draft_tokens,
            draft_token_acc_rate,
        };

        rayon::spawn(
            move || loop {
                let mut resps = responder
                    .await_responses(std::time::Duration::from_millis(1))
                    .unwrap();

                if let Some(x) = draft_responder.as_mut() {
                    resps.append(&mut x
                    .await_responses(std::time::Duration::from_millis(1))
                    .unwrap())
                };

                if resps.len() == 0 {
                    continue;
                }

                let mut exec = AsyncExecutor::lock();
                for resp in resps {
                    let req_id = resp.req_id;
                    if let Some(client_req_id) = exec.req_to_client.get(&req_id) {
                        let client_req_id = *client_req_id;
                        let rd = exec.req_data.get_mut(&client_req_id).unwrap();
                        let is_req_final = resp.is_req_final;
                        let idx = resp.sequence_idx as usize;

                        let mut r = StepResults {
                            response: resp,
                            logs: std::mem::take(&mut rd.logs),
                            final_llg: None,
                        };
                        if rd.llgs.len() > 0 && r.response.finish_reason.is_some() {
                            r.final_llg = std::mem::take(&mut rd.llgs[idx]);
                        }
                        if rd.tx.send(r).is_err() {
                            log::warn!("connection dropped; req={}", req_id);
                            let _ = exec.cancel_request(req_id);
                        } else if is_req_final {
                            // no more data coming from here
                            exec.drop_request_data(req_id);
                        }
                    } else {
                        log::warn!("Response for unknown request: {:?}", req_id);
                        log::debug!("Tokens for unknown request {:?}: {:?}", req_id, resp.tokens);
                        let _ = exec.executor.cancel_request(req_id);
                    }
                }
            }
        );

        Ok((res, tok_env, chat_builder))
    }

    pub fn can_enqueue_request(&self) -> bool {
        self.executor.can_enqueue_request() && self.draft_executor.as_ref().map_or(true, |ex| ex.can_enqueue_request())
    }

    pub fn add_draft_request(
        &mut self,
        init: &RequestInit,
        prompt_params: Option<Arc<PyPromptParams>>,
        llgs: Vec<Box<Constraint>>,
    ) -> Result<(ReqId, UnboundedReceiver<StepResults>)> {
        debug_assert!(self.draft_executor.is_some());
        self.add_request_to_executor(
            init,
            prompt_params,
            llgs,
            true
        )
    }

    pub fn add_request(
        &mut self,
        init: &RequestInit,
        prompt_params: Option<Arc<PyPromptParams>>,
        llgs: Vec<Box<Constraint>>,
    ) -> Result<(ReqId, UnboundedReceiver<StepResults>)> {
        self.add_request_to_executor(init, prompt_params, llgs, false)
    }

    fn add_request_to_executor(
        &mut self,
        init: &RequestInit,
        prompt_params: Option<Arc<PyPromptParams>>,
        llgs: Vec<Box<Constraint>>,
        use_draft_model: bool
    ) -> Result<(ReqId, UnboundedReceiver<StepResults>)> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        ensure!(llgs.len() == 0 || llgs.len() == init.params.num_return_sequences as usize);

        let client_req_id = init.client_req_id;
        let prompt_len = init.tokens.len();
        let is_run = init.is_run;

        let pp = prompt_params.as_ref().map(|p| &p.tlc_prompt_params);

        let req_id = if use_draft_model {
            self.draft_executor.as_mut().expect("msg").enqueue_request(init, pp)?
        } else {
            self.executor.enqueue_request(init, pp)?
        };

        self.req_data.insert(
            client_req_id,
            ReqData {
                req_id,
                client_req_id,
                tx,
                llgs: llgs.into_iter().map(Some).collect(),
                llg_infos: vec![],
                prompt_len,
                min_p: init.params.min_p,
                logs: String::new(),
                is_run,
                _prompt_params: prompt_params,
            },
        );
        self.req_to_client.insert(req_id, client_req_id);

        Ok((req_id, rx))
    }
}
