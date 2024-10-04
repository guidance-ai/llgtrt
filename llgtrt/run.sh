#!/bin/bash

# ENGINE=/root/trt-cache/Meta-Llama-3.1-8B-Instruct-engine/
ENGINE=/root/trt-cache/llama-8b-1tp
# ENGINE=/root/trt-cache/engine-llama3.1-70b-4tp
# ENGINE=/root/trt-cache/llama-8b-4tp

set -e

make -j -C ../trtllm-c/build
cargo build --release
RUST_BACKTRACE=1 \
LLGTRT_BIN=../target/release/llgtrt \
    ../scripts/launch-llgtrt.sh \
    $ENGINE \
    --chat-config chat_config/llama3.json \
    "$@"
