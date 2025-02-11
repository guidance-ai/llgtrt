use crate::chat::ChatParams;
use crate::config::{CliConfig, LlgTrtConfig};
use crate::routes::openai::{ChatCompletionMessageContentPart, ChatCompletionMessageParams};
use crate::routes::RequestInput;
use anyhow::Result;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi::c_str;
use pyo3::types::{PyDict, PyList, PyString};
use pyo3::{prelude::*, BoundObject};
use serde_json::Value;
use std::ffi::CStr;
use std::fmt::Display;
use std::sync::Arc;
use toktrie::{TokEnv, TokTrie, TokenId};
use trtllm_rs::{TlcDataType, TlcPromptParams, TlcShape, TlcTensor};

pub struct PyPromptParams {
    pub tlc_prompt_params: TlcPromptParams,
    // holding this makes sure tlc_prompt_params memory is not dropped
    pub tensor_ref: pyo3::PyObject,
}
unsafe impl Send for PyPromptParams {}
unsafe impl Sync for PyPromptParams {}

pub struct PyState {
    pub enabled: bool,
    file_content: String,
    file_name: String,
    plugin_ref: Option<PyObject>,
}

#[pyclass]
pub struct PluginInit {
    #[pyo3(get)]
    pub engine_dir: String,
    #[pyo3(get)]
    pub tokenizer_dir: String,
    #[pyo3(get)]
    pub chat_template: String,
    #[pyo3(get)]
    pub bos_token: Option<String>,
    #[pyo3(get)]
    pub eos_token: Option<String>,

    #[pyo3(get)]
    pub visual_engine_dir: String,
    #[pyo3(get)]
    pub hf_model_dir: String,
    #[pyo3(get)]
    pub arguments: PyObject,
}

#[pyfunction]
fn torch_dtype(tp: &str) -> PyResult<i32> {
    let r = match tp {
        "torch.float32" => TlcDataType::TLC_DT_F32,
        "torch.float16" => TlcDataType::TLC_DT_F16,
        "torch.int8" => TlcDataType::TLC_DT_I8,
        "torch.int32" => TlcDataType::TLC_DT_I32,
        "torch.bool" => TlcDataType::TLC_DT_BOOL,
        "torch.uint8" => TlcDataType::TLC_DT_U8,
        "torch.float8" => TlcDataType::TLC_DT_F8,
        "torch.bfloat16" => TlcDataType::TLC_DT_BF16,
        "torch.int64" => TlcDataType::TLC_DT_I64,
        "torch.int4" => TlcDataType::TLC_DT_I4,
        _ => return Err(PyRuntimeError::new_err("Unknown torch dtype")),
    };
    Ok(r as i32)
}

#[pymodule]
fn llgtrt_native(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(torch_dtype, m)?)?;
    m.add_class::<PluginInit>()?;
    Ok(())
}

fn rt_error(e: impl Display) -> PyErr {
    PyRuntimeError::new_err(format!("{e}"))
}

impl PyState {
    pub fn test(&self) {
        let r = self
            .run_input_processor(ChatParams {
                messages: &vec![ChatCompletionMessageParams::User {
                    content: ChatCompletionMessageContentPart::Text("Hello world!".to_string()),
                    name: None,
                }],
                tools: &vec![],
                json_schema: None,
            })
            .expect("Failed to run input processor");
        log::warn!("--test-py result: {:?}", r.prompt);
    }

    pub fn run_input_processor(&self, chat_params: ChatParams) -> Result<RequestInput> {
        let chat_params = serde_json::to_string(&chat_params).unwrap();
        Python::with_gil(|py| {
            let loc = self.mk_locals(py);
            loc.set_item("chat_params", chat_params).unwrap();
            let r = self.eval(py, "plugin._process_input(chat_params)", Some(&loc))?;
            let r2 = r
                .downcast::<PyDict>()
                .map_err(|e| PyRuntimeError::new_err(format!("Expected dict, got {}", e)))?;

            let mut pp = TlcPromptParams::default();

            pp.prompt_table = self.get_tensor(&r2, "prompt_table")?;
            pp.input_token_extra_ids = self.get_tensor(&r2, "input_token_extra_ids")?;
            pp.mrope_rotary_sin_cos = self.get_tensor(&r2, "mrope_rotary_sin_cos")?;
            pp.mrope_position_deltas =
                self.get_i32(&r2, "mrope_position_deltas", pp.mrope_position_deltas)?;
            pp.skip_cross_attn_blocks = self.get_tensor(&r2, "skip_cross_attn_blocks")?;
            pp.encoder_input_features = self.get_tensor(&r2, "encoder_input_features")?;
            pp.encoder_output_length =
                self.get_i32(&r2, "encoder_output_length", pp.encoder_output_length)?;
            pp.cross_attention_masks = self.get_tensor(&r2, "cross_attention_masks")?;
            pp.input_position_ids = self.get_tensor(&r2, "input_position_ids")?;

            Ok(RequestInput {
                tokens: self.get_field(r2, "tokens")?.extract()?,
                prompt: self.get_field(r2, "prompt")?.extract()?,
                prompt_params: Some(Arc::new(PyPromptParams {
                    tlc_prompt_params: pp,
                    tensor_ref: r.into(),
                })),
            })
        })
    }

    fn get_i32<'py>(&self, dict: &Bound<'py, PyDict>, field: &str, defl: i32) -> PyResult<i32> {
        let t: Option<i32> = self.get_field(dict, field)?.extract()?;
        Ok(t.unwrap_or(defl))
    }

    fn get_tensor<'py>(&self, dict: &Bound<'py, PyDict>, field: &str) -> PyResult<TlcTensor> {
        let t: Option<(PyObject, u32, usize, Vec<i64>)> = self.get_field(dict, field)?.extract()?;
        if let Some((_, dtype, addr, shape)) = t {
            let dtype = TlcDataType::try_from(dtype).map_err(rt_error)?;
            // some sanity checks
            if addr == 0 {
                return Err(PyRuntimeError::new_err("NULL tensor address"));
            }
            if addr % 8 != 0 {
                return Err(PyRuntimeError::new_err("Tensor address not aligned"));
            }
            Ok(TlcTensor {
                data_type: dtype,
                shape: TlcShape::from_slice(&shape),
                data_ptr: addr as *const _,
            })
        } else {
            Ok(TlcTensor::default())
        }
    }

    fn get_field<'py>(
        &self,
        dict: &Bound<'py, PyDict>,
        field: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        dict.get_item(field)?
            .ok_or_else(|| PyRuntimeError::new_err(format!("Field {} not found", field)))
    }

    fn add_traceback(&self, py: Python<'_>, e: PyErr) -> PyErr {
        let traceback = e
            .traceback(py)
            .map(|tb| format!("{}", tb.format().unwrap_or_default()))
            .unwrap_or_else(|| "No traceback available".to_string());
        // .replace(
        //     "File \"<string>\", line ",
        //     &format!("File \"{}\", line ", self.file_name),
        // );
        PyRuntimeError::new_err(format!("{}\n{}", e, traceback))
    }

    fn mk_locals<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        if let Some(plugin) = &self.plugin_ref {
            locals.set_item("plugin", plugin).unwrap();
        }
        locals
    }

    fn eval<'py>(
        &self,
        py: Python<'py>,
        code: &str,
        locals: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut code = code.to_string();
        code.push('\0');
        let call = CStr::from_bytes_with_nul(code.as_bytes()).map_err(rt_error)?;
        py.eval(call, None, locals)
            .map_err(|e| self.add_traceback(py, e))
    }
}

fn token_name(trie: &TokTrie, tok: TokenId) -> String {
    let b = trie.token(tok);
    if b.len() > 0 && b[0] == TokTrie::SPECIAL_TOKEN_MARKER {
        String::from_utf8_lossy(&b[1..]).to_string()
    } else {
        String::from_utf8_lossy(&b[0..]).to_string()
    }
}

pub fn init(tok_env: &TokEnv, cli_config: &CliConfig, cfg: &LlgTrtConfig) -> Result<PyState> {
    let mut state = if let Some(inp) = &cfg.py.input_processor {
        log::info!("Reading Python script from {}", inp);
        PyState {
            enabled: true,
            file_content: std::fs::read_to_string(inp)?,
            file_name: inp.to_string(),
            plugin_ref: None,
        }
    } else {
        log::info!("No Python script provided");
        PyState {
            enabled: false,
            file_content: "".to_string(),
            file_name: "".to_string(),
            plugin_ref: None,
        }
    };

    if !state.enabled {
        return Ok(state);
    }

    state.file_content.push('\0');
    state.file_name.push('\0');

    let llgtrt_base = c_str!(include_str!("../py/llgtrt_base.py"));

    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let trie = tok_env.tok_trie();
        let plugin_init = PluginInit {
            engine_dir: cli_config.engine.clone(),
            tokenizer_dir: cli_config
                .tokenizer
                .as_ref()
                .unwrap_or(&cli_config.engine)
                .to_string(),
            chat_template: cfg.tokenizer.chat_template.clone().unwrap_or_default(),
            bos_token: trie.info().tok_bos.map(|t| token_name(trie, t)),
            eos_token: Some(token_name(trie, trie.info().tok_eos)),

            // py
            hf_model_dir: cfg.py.hf_model_dir.clone().unwrap_or_default(),
            visual_engine_dir: cfg.py.visual_engine_dir.clone().unwrap_or_default(),
            arguments: serde_value_to_py(py, &cfg.py.arguments).unwrap(),
        };

        let native_mod = PyModule::new(py, "llgtrt_native")?;
        llgtrt_native(py, &native_mod)?;

        // Register the module in Python's sys.modules
        let sys = py.import("sys")?;
        let sys_modules = sys.getattr("modules")?;
        let sys_modules = sys_modules.downcast::<PyDict>()?;
        sys_modules.set_item("llgtrt_native", native_mod)?;

        let base_module = PyModule::from_code(py, &llgtrt_base, c"llgtrt_base.py", c"llgtrt_base")
            .map_err(|e| state.add_traceback(py, e))?;
        sys_modules.set_item("llgtrt_base", base_module)?;

        let scr = CStr::from_bytes_with_nul(state.file_content.as_bytes()).map_err(rt_error)?;
        let main_file = CStr::from_bytes_with_nul(state.file_name.as_bytes()).map_err(rt_error)?;
        let main_module = PyModule::from_code(py, scr, main_file, c"_plugin")
            .map_err(|e| state.add_traceback(py, e))?;
        sys_modules.set_item("_plugin", main_module.clone())?;
        // we don't add it anywhere

        let locals = PyDict::new(py);
        locals.set_item("init", plugin_init)?;
        locals.set_item("_plugin", main_module)?;
        log::debug!("Executing plugin init");
        let result = state.eval(py, "_plugin.Plugin(init)", Some(&locals))?;
        log::info!("Plugin init result: {:?}", result);

        state.plugin_ref = Some(result.into());

        Ok(state)
    })
    .map_err(|e: PyErr| anyhow::anyhow!("Python error: {}", e))
}

fn serde_value_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<PyObject> {
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.into_pyobject(py).unwrap().into_any().unbind()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py).unwrap().into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py).unwrap().into_any().unbind())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err("Invalid number"))
            }
        }
        Value::String(s) => Ok(PyString::new(py, s).into()),
        Value::Array(arr) => {
            let py_list = PyList::empty(py);
            for item in arr {
                py_list.append(serde_value_to_py(py, item)?)?;
            }
            Ok(py_list.into())
        }
        Value::Object(obj) => {
            let py_dict = PyDict::new(py);
            for (key, val) in obj {
                py_dict.set_item(key, serde_value_to_py(py, val)?)?;
            }
            Ok(py_dict.into())
        }
    }
}
