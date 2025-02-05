use crate::config::LlgTrtPyConfig;
use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CStr;

/// A simple Rust function exposed to Python
#[pyfunction]
fn rust_function(x: i32, y: i32) -> i32 {
    x + y
}

#[pymodule]
fn rust_api(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_function, m)?)?;
    Ok(())
}

pub fn init(cfg: &LlgTrtPyConfig) -> Result<()> {
    let mut scr = if let Some(inp) = &cfg.input_processor {
        log::info!("Reading Python script from {}", inp);
        std::fs::read_to_string(inp)?
    } else {
        log::info!("No Python script provided");
        return Ok(());
    };
    scr.push('\0');

    // Initialize Python
    Python::with_gil(|py| {
        // Create a new Python module for Rust functions
        let modu = PyModule::new(py, "rust_api")?;
        rust_api(py, &modu)?;

        // Register the module in Python's sys.modules
        let sys = py.import("sys")?;
        let sys_modules = sys.getattr("modules")?;
        let sys_modules = sys_modules
            .downcast::<PyDict>()
            .map_err(|_| anyhow::anyhow!("sys.modules is not a dictionary"))?;
        sys_modules.set_item("rust_api", modu)?;

        // Create a global scope and inject the Rust API module
        let globals = PyDict::new(py);
        globals.set_item("__name__", "__main__")?;

        // Execute the Python script
        let scr = CStr::from_bytes_with_nul(scr.as_bytes())?;
        py.run(scr, Some(&globals), None)?;

        // Assume the Python script defines a `get_data()` function
        let locals = PyDict::new(py);
        let call = CStr::from_bytes_with_nul(b"result = get_data()\0")?;
        py.run(call, Some(&globals), Some(&locals))?;

        // Retrieve the dictionary returned from Python
        let result0 = locals
            .get_item("result")?
            .ok_or_else(|| anyhow::anyhow!("Missing result variable?!"))?;
        let result = result0
            .downcast::<PyDict>()
            .map_err(|_| anyhow::anyhow!("Expected a dictionary"))?;

        // Print the result
        println!("Python returned: {:?}", result);

        Ok(())
    })
}
