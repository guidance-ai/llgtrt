#!/bin/bash

USE_CXX11_ABI=0

while test $# -gt 0; do
    case "$1" in
        --clean)
            rm -rf trtllm-c/build
            rm -rf target/release/* 2>/dev/null || :
            shift
            ;;
        --cxx11abi)
            if [[ -n "$2" && "$2" =~ ^[01]$ ]]; then
                USE_CXX11_ABI="$2"
                shift 2
            else
                echo "Error: --cxx11abi requires an argument (0 or 1)."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

set -e
if test -f TensorRT-LLM/README.md -a -f llguidance/README.md ; then
    :
else
    echo "Cloning submodules"
    git submodule update --init
fi

mkdir -p trtllm-c/build
cd trtllm-c/build
cmake -DUSE_CXX11_ABI=$USE_CXX11_ABI ..
make -j
cd ../../llgtrt
export RUSTC_LOG=rustc_codegen_ssa::back::link=info
cargo build --release
