use crate::chat::ChatParams;
use crate::config::{CliConfig, LlgTrtConfig};
use crate::routes::openai::{ChatCompletionMessageContentPart, ChatCompletionMessageParams};
use crate::routes::RequestInput;
use anyhow::Result;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CStr;
use std::fmt::Display;
use std::sync::Arc;
use trtllm_rs::{TlcPromptParams, TlcTensor};

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
    pub tokenizer_folder: String,
    #[pyo3(get)]
    pub chat_template: String,
    #[pyo3(get)]
    pub hf_model_dir: String,
}

#[pyfunction]
fn rust_function(x: i32, y: i32) -> i32 {
    x + y
}

#[pymodule]
fn llgtrt_native(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_function, m)?)?;
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

    fn get_tensor<'py>(&self, dict: &Bound<'py, PyDict>, field: &str) -> PyResult<TlcTensor> {
        let t: Option<(PyObject, usize, Vec<usize>)> = self.get_field(dict, field)?.extract()?;
        if let Some((_, _dtype, _shape)) = t {
            Ok(TlcTensor::default())
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

pub fn init(cli_config: &CliConfig, cfg: &LlgTrtConfig) -> Result<PyState> {
    let plugin_init = PluginInit {
        tokenizer_folder: cli_config
            .tokenizer
            .as_ref()
            .unwrap_or(&cli_config.engine)
            .to_string(),
        chat_template: cfg.tokenizer.chat_template.clone().unwrap_or_default(),
        hf_model_dir: cfg.py.hf_model_dir.clone().unwrap_or_default(),
    };

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
        let main_fn = CStr::from_bytes_with_nul(state.file_name.as_bytes()).map_err(rt_error)?;
        let main_module = PyModule::from_code(py, scr, main_fn, c"_plugin")
            .map_err(|e| state.add_traceback(py, e))?;
        sys_modules.set_item("_plugin", main_module.clone())?;
        // we don't add it anywhere

        let locals = PyDict::new(py);
        locals.set_item("init", plugin_init)?;
        locals.set_item("_plugin", main_module)?;
        let result = state.eval(py, "_plugin.Plugin(init)", Some(&locals))?;
        state.plugin_ref = Some(result.into());

        Ok(state)
    })
    .map_err(|e: PyErr| anyhow::anyhow!("Python error: {}", e))
}
