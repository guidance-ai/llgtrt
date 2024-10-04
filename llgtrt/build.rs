use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rustc-link-lib=trtllm_c");
    let build_dir = format!("{}/../trtllm-c/build", crate_dir);
    println!("cargo:rustc-link-search=native={}", build_dir);

    for libs_path in &[
        "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs",
        "/usr/local/cuda/lib64",
    ] {
        println!("cargo:rustc-link-search=native={}", libs_path);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libs_path);
    }
    println!("cargo:rustc-link-lib=tensorrt_llm");
    println!("cargo:rustc-link-lib=nvinfer_plugin_tensorrt_llm");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
}
