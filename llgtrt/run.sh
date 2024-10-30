#!/bin/bash

if [ -z "$1" ] ; then
    ENGINE=${ENGINE:-/root/trt-cache/llama-8b-1tp}
else
    ENGINE="$1"
    shift
fi

set -e

make -j -C ../trtllm-c/build
cargo build --release
RUST_BACKTRACE=1 \
LLGTRT_BIN=../target/release/llgtrt \
    ../scripts/launch-llgtrt.sh \
    $ENGINE \
    "$@"
