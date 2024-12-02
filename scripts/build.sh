#!/bin/bash


while test $# -gt 0; do
    case "$1" in
        --clean)
            rm -rf trtllm-c/build
            rm -rf target/release/* 2>/dev/null || :
            shift
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
cmake ..
make -j
cd ../../llgtrt
cargo build --release
