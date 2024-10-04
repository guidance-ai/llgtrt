extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rustc-link-lib=trtllm_c");
    // let build_dir = format!("{}/../trtllm-c/build", crate_dir);
    // println!("cargo:rustc-link-search=native={}", build_dir);
    // println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir);

    println!("cargo:rerun-if-changed={}/../trtllm-c/tlc.h", crate_dir);
    println!("cargo:rerun-if-changed={}/../trtllm-c/build/libtrtllm_c.a", crate_dir);

    let bindings = bindgen::Builder::default()
        .header("../trtllm-c/tlc.h")
        .generate()
        .expect("Unable to generate bindings");

    // save bindings to string and replace "Copy, Clone" with "Clone"
    let bindings = bindings
        .to_string()
        .replace("#[derive(Debug, Copy, Clone)]", "#[derive(Debug, Clone)]")
        .replace(
            "#[derive(Debug, Clone)]\npub struct TlcEngineParams {",
            "#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct TlcEngineParams {",
        );
    let bindings = format!("use serde::{{Deserialize, Serialize}};\n{}", bindings);

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Write the bindings to $OUT_DIR/bindings.rs
    std::fs::write(out_path.join("bindings.rs"), bindings).expect("Couldn't write bindings!");
}
