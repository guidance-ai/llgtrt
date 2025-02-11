use crate::{
    ffi::{self, TlcTensor},
    TlcDataType, TlcLogitsEntry,
};
use anyhow::{ensure, Result};
use std::{
    ffi::{c_void, CStr, CString},
    fmt::Display,
    hash::Hash,
    ptr,
    sync::atomic::AtomicU32,
    time::Duration,
};

pub type TokenId = u32;

#[derive(Debug, Clone, Default)]
pub struct ExecutorInit {
    pub engine_path: String,
    pub logits_callback: ffi::TlcLogitsPostProcessor,
    pub trt_params: ffi::TlcEngineParams,
}

#[derive(Debug, Clone)]
pub struct ResponseChunk {
    pub req_id: ReqId,
    pub sequence_idx: u32,
    pub finish_reason: Option<FinishReason>,
    pub error: Option<String>,
    pub tokens: Vec<TokenId>,
    pub logprobs: Option<Vec<Vec<(TokenId, f32)>>>,
    pub is_req_final: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The request finished because the end id was generated.
    EosToken = ffi::TlcFinishReason_FINISH_REASON_END_ID as isize,

    /// The request finished because a stop word was generated.
    StopWords = ffi::TlcFinishReason_FINISH_REASON_STOP_WORDS as isize,

    /// The request finished because the maximum number of tokens was reached.
    Length = ffi::TlcFinishReason_FINISH_REASON_LENGTH as isize,

    /// Something else.
    Unknown = 100 as isize,
}

pub type RequestParams = ffi::TlcRequestParams;

impl Default for ffi::TlcPromptParams {
    fn default() -> Self {
        ffi::TlcPromptParams {
            prompt_table: TlcTensor::default(),
            input_token_extra_ids: TlcTensor::default(),
            mrope_rotary_sin_cos: TlcTensor::default(),
            mrope_position_deltas: 0,
            skip_cross_attn_blocks: TlcTensor::default(),
            encoder_input_features: TlcTensor::default(),
            encoder_output_length: -1,
            cross_attention_masks: TlcTensor::default(),
            input_position_ids: TlcTensor::default(),
        }
    }
}

impl Default for RequestParams {
    fn default() -> Self {
        ffi::TlcRequestParams {
            streaming: true,
            max_new_tokens: 10,
            num_return_sequences: 1,
            temperature: f32::NAN,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            top_k: 0,
            priority: 0.5, // kDefaultPriority
            min_tokens: 1,
            eos_token_id: u32::MAX,
            seed: u64::MAX,
            use_logits_post_processor: false,
            logprobs: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ReqId(ffi::TlcReqId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ClientReqId(ffi::TlcClientId);

impl Display for ReqId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "req{}", self.0)
    }
}

impl Display for ClientReqId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "creq{}", self.0)
    }
}

impl ClientReqId {
    pub fn new(id: u64) -> Self {
        assert!(id > 0);
        ClientReqId(id)
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub size: Vec<i64>,
    pub dtype: TlcDataType,
    pub data: Vec<u8>,
}

impl Default for Tensor {
    fn default() -> Self {
        Tensor {
            size: vec![],
            dtype: TlcDataType::TLC_DT_F32,
            data: vec![],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LoraParams {
    pub lora_id: u64,
    pub weights: Option<Tensor>,
    pub config: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct RequestInit {
    pub tokens: Vec<TokenId>,
    pub client_req_id: ClientReqId,
    pub is_run: bool,
    pub params: RequestParams,
    pub lora_params: Option<LoraParams>,
}

unsafe impl Send for ffi::TlcPromptParams {}

pub struct Executor {
    inner: *mut ffi::TlcExecutor,
}
unsafe impl Send for Executor {}

pub struct Responder {
    inner: *mut ffi::TlcExecutor,
}
unsafe impl Send for Responder {}

pub struct MaskAllocator {
    base_ptr: *mut u8,
    max_batch_size: usize,
    mask_stride: usize,
    num_bytes: usize,
    curr_offset: AtomicU32,
    mask_fraction_ptr: *mut f32,
}
unsafe impl Send for MaskAllocator {}

impl MaskAllocator {
    pub fn new(n_vocab: usize, max_batch_size: usize) -> Self {
        let mask_stride = (n_vocab + 31) / 32 * 4;
        let base_ptr =
            unsafe { ffi::tlc_alloc_logit_data(mask_stride as i32, max_batch_size as i32) };
        assert!(base_ptr != std::ptr::null_mut());
        MaskAllocator {
            base_ptr: base_ptr as *mut _,
            max_batch_size,
            mask_stride,
            curr_offset: AtomicU32::new(0),
            num_bytes: mask_stride * max_batch_size,
            mask_fraction_ptr: unsafe { ffi::tlc_mask_fraction_ptr() },
        }
    }

    pub fn mask_fractions(&self, n: usize) -> Vec<f32> {
        assert!(n <= self.max_batch_size);
        let slice = unsafe { std::slice::from_raw_parts(self.mask_fraction_ptr, n) };
        slice.to_vec()
    }

    pub fn reset(&self) {
        self.curr_offset
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn allocate(&self) -> &mut [u32] {
        let offset = self.curr_offset.fetch_add(
            self.mask_stride as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        assert!((offset + self.mask_stride as u32) < self.num_bytes as u32);
        unsafe {
            let ptr = self.base_ptr.add(offset as usize);
            std::slice::from_raw_parts_mut(ptr as *mut u32, self.mask_stride / 4)
        }
    }
}

impl Default for ffi::TlcEngineParams {
    fn default() -> Self {
        let mut init: ffi::TlcInitParams = unsafe { std::mem::zeroed() };
        unsafe { ffi::tlc_default_init_params(&mut init) };
        init.engine_params
    }
}

impl Default for ffi::TlcShape {
    fn default() -> Self {
        ffi::TlcShape {
            dims: [0; ffi::TLC_MAX_SHAPE as usize],
            num_dims: 0,
        }
    }
}

impl ffi::TlcShape {
    pub fn from_slice(shape: &[i64]) -> Self {
        assert!(shape.len() <= ffi::TLC_MAX_SHAPE as usize);
        let mut r = ffi::TlcShape::default();
        r.num_dims = shape.len();
        for (i, &d) in shape.iter().enumerate() {
            r.dims[i] = d;
        }
        r
    }
}

impl Default for ffi::TlcTensor {
    fn default() -> Self {
        ffi::TlcTensor {
            data_type: TlcDataType::TLC_DT_F32,
            data_ptr: ptr::null(),
            shape: ffi::TlcShape::default(),
        }
    }
}

impl Default for ffi::TlcLoraParams {
    fn default() -> Self {
        ffi::TlcLoraParams {
            lora_id: 0,
            weights: ffi::TlcTensor::default(),
            config: ffi::TlcTensor::default(),
        }
    }
}

impl Tensor {
    pub fn as_tlc_tensor(&self) -> ffi::TlcTensor {
        let tensor = self;
        let ffi_shape = ffi::TlcShape::from_slice(&tensor.size);

        // Get the values in pure vector form
        let data_ptr = tensor.data.as_ptr();
        let void_data_ptr = data_ptr as *const c_void;

        ffi::TlcTensor {
            shape: ffi_shape,
            data_ptr: void_data_ptr,
            data_type: tensor.dtype,
        }
    }
}

impl Executor {
    pub fn new(init: ExecutorInit) -> Result<(Executor, Responder)> {
        let cstr = CString::new(init.engine_path).unwrap();
        let params = ffi::TlcInitParams {
            engine_path: cstr.as_ptr(),
            logits_post_processor: init.logits_callback,
            engine_params: init.trt_params,
        };
        let mut inner = std::ptr::null_mut();
        let err = unsafe { ffi::tlc_init(&params, &mut inner) };
        let r = (Executor { inner }, Responder { inner });
        map_err(err, r, "tlc_init")
    }

    pub fn can_enqueue_request(&self) -> bool {
        unsafe { ffi::tlc_can_enqueue_request(self.inner) }
    }

    pub fn check_mpi(&self) {
        if !self.can_enqueue_request() {
            unsafe { ffi::tlc_shutdown(self.inner) };
            std::process::exit(0);
        }
    }

    pub fn enqueue_request(
        &mut self,
        init: &RequestInit,
        prompt_params: Option<&ffi::TlcPromptParams>,
    ) -> Result<ReqId> {
        ensure!(self.can_enqueue_request(), "Cannot enqueue request");
        ensure!(
            init.tokens.len() > 0,
            "Request must have at least one token"
        );

        let mut arg = ffi::TlcRequest {
            tokens: init.tokens.as_ptr() as *mut i32,
            num_tokens: init.tokens.len() as u32,
            params: init.params.clone(),
            client_req_id: init.client_req_id.0,
            lora_params: ffi::TlcLoraParams::default(),
            prompt_params: ffi::TlcPromptParams::default(),
        };

        if let Some(pp) = prompt_params {
            arg.prompt_params = pp.clone();
        }

        if let Some(lora_params) = &init.lora_params {
            let mut lp = ffi::TlcLoraParams {
                lora_id: lora_params.lora_id,
                weights: ffi::TlcTensor::default(),
                config: ffi::TlcTensor::default(),
            };
            if let Some(weights) = &lora_params.weights {
                lp.weights = weights.as_tlc_tensor();
            }
            if let Some(config) = &lora_params.config {
                lp.config = config.as_tlc_tensor();
            }
            arg.lora_params = lp;
        }

        let mut req_id = 0;
        let err = unsafe { ffi::tlc_enqueue_request(self.inner, &arg, &mut req_id) };
        map_err(err, ReqId(req_id), "tlc_enqueue_request")
    }

    pub fn cancel_request(&mut self, req_id: ReqId) -> Result<()> {
        let err = unsafe { ffi::tlc_cancel_request(self.inner, req_id.0) };
        map_err(err, (), "tlc_cancel_request")
    }
}

impl TlcLogitsEntry {
    #[inline(always)]
    pub fn req_id(&self) -> ReqId {
        ReqId(self._req_id)
    }
    #[inline(always)]
    pub fn client_req_id(&self) -> ClientReqId {
        ClientReqId(self._client_req_id)
    }
    #[inline(always)]
    pub unsafe fn tokens(&self) -> &[TokenId] {
        std::slice::from_raw_parts(self._tokens as *const _, self._num_tokens as usize)
    }
}

impl Display for TlcLogitsEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}, {} toks)",
            self.req_id(),
            self.client_req_id(),
            self._num_tokens
        )
    }
}

impl Responder {
    pub fn await_responses(&mut self, timeout: Duration) -> Result<Vec<ResponseChunk>> {
        let mut responses: *const ffi::TlcResponse = std::ptr::null();
        let mut num_responses = 0;

        let err = unsafe {
            ffi::tlc_await_responses(
                self.inner,
                timeout.as_millis().try_into().unwrap_or(u32::MAX),
                &mut responses,
                &mut num_responses,
            )
        };

        if let Err(e) = map_err(err, (), "tlc_await_responses") {
            Err(e)
        } else {
            Ok((0..num_responses)
                .map(|i| {
                    let resp = unsafe { &*responses.add(i as usize) };
                    let finish_reason = if resp.is_seq_final {
                        match resp.finish_reason {
                            ffi::TlcFinishReason_FINISH_REASON_END_ID => {
                                Some(FinishReason::EosToken)
                            }
                            ffi::TlcFinishReason_FINISH_REASON_STOP_WORDS => {
                                Some(FinishReason::StopWords)
                            }
                            ffi::TlcFinishReason_FINISH_REASON_LENGTH => Some(FinishReason::Length),
                            _ => Some(FinishReason::Unknown),
                        }
                    } else {
                        None
                    };
                    let tokens = unsafe {
                        std::slice::from_raw_parts(
                            resp.tokens as *const u32,
                            resp.num_tokens as usize,
                        )
                    }
                    .to_vec();

                    let logprobs = if resp.num_logprobs == 0 {
                        None
                    } else {
                        let logprob_data = unsafe {
                            std::slice::from_raw_parts(resp.logprobs, resp.num_logprobs as usize)
                        }
                        .to_vec();

                        assert!(logprob_data.len() == tokens.len());
                        Some(
                            tokens
                                .iter()
                                .zip(logprob_data.iter())
                                .map(|(t, p)| vec![(*t, *p)])
                                .collect(),
                        )
                    };

                    ResponseChunk {
                        req_id: ReqId(resp.req_id),
                        sequence_idx: resp.sequence_idx,
                        finish_reason,
                        is_req_final: resp.is_req_final,
                        error: if resp.error.is_null() {
                            None
                        } else {
                            Some(
                                unsafe { CStr::from_ptr(resp.error) }
                                    .to_str()
                                    .unwrap_or("Invalid UTF-8 in seq error")
                                    .to_owned(),
                            )
                        },
                        logprobs,
                        tokens,
                    }
                })
                .collect())
        }
    }
}

fn map_err<T>(err: *const i8, t: T, msg: &str) -> Result<T> {
    if !err.is_null() {
        let c_str = unsafe { CStr::from_ptr(err) };
        let rust_string = c_str
            .to_str()
            .unwrap_or("Invalid UTF-8 in error")
            .to_owned();
        let r = Err(anyhow::anyhow!("{}: {}", msg, rust_string));
        unsafe { libc::free(err as *mut _) }
        r
    } else {
        Ok(t)
    }
}

impl TryFrom<u32> for TlcDataType {
    type Error = &'static str;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value == TlcDataType::TLC_DT_F32 as u32 {
            Ok(TlcDataType::TLC_DT_F32)
        } else if value == TlcDataType::TLC_DT_F16 as u32 {
            Ok(TlcDataType::TLC_DT_F16)
        } else if value == TlcDataType::TLC_DT_I8 as u32 {
            Ok(TlcDataType::TLC_DT_I8)
        } else if value == TlcDataType::TLC_DT_I32 as u32 {
            Ok(TlcDataType::TLC_DT_I32)
        } else if value == TlcDataType::TLC_DT_BOOL as u32 {
            Ok(TlcDataType::TLC_DT_BOOL)
        } else if value == TlcDataType::TLC_DT_U8 as u32 {
            Ok(TlcDataType::TLC_DT_U8)
        } else if value == TlcDataType::TLC_DT_F8 as u32 {
            Ok(TlcDataType::TLC_DT_F8)
        } else if value == TlcDataType::TLC_DT_BF16 as u32 {
            Ok(TlcDataType::TLC_DT_BF16)
        } else if value == TlcDataType::TLC_DT_I64 as u32 {
            Ok(TlcDataType::TLC_DT_I64)
        } else if value == TlcDataType::TLC_DT_I4 as u32 {
            Ok(TlcDataType::TLC_DT_I4)
        } else {
            Err("Invalid TlcDataType")
        }
    }
}
