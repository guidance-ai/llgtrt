use crate::config::{CliConfig, LlgTrtConfig};
use crate::routes::RequestInput;
use anyhow::Result;
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CStr;
use std::fmt::Display;

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
    pub fn run_input_processor(&self) -> RequestInput {
        RequestInput {
            tokens: vec![],
            prompt: "".to_string(),
        }
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
