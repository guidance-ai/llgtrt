#!/bin/sh

set -e
cd $(dirname $0)/..
SELF=./scripts/trtbld.sh

CACHE=${CACHE:-/root/trt-cache}
MODEL=${MODEL:-Meta-Llama-3.1-8B-Instruct}
MODEL_TYPE=${MODEL_TYPE:-llama}
LLAMA_EXAMPLE=$(pwd)/TensorRT-LLM/examples/$MODEL_TYPE
MODEL_SRC=$CACHE/$MODEL-hf

CKPT=$CACHE/$MODEL-ckpt
ENGINE_DIR=$CACHE/$MODEL-engine

TP_SIZE=${TP_SIZE:-1}

set -x

case "$1" in
    all)
        $SELF clean
        $SELF convert
        $SELF build
        ;;

    clean)
        rm -rf $CKPT $ENGINE_DIR
        ;;

    convert)
        python3 $LLAMA_EXAMPLE/convert_checkpoint.py \
            --dtype bfloat16 \
            --model_dir $MODEL_SRC \
            --output_dir $CKPT \
            --tp_size $TP_SIZE
        ;;
    
    build)
        trtllm-build --checkpoint_dir $CKPT \
            --gemm_plugin bfloat16 \
            --output_dir $ENGINE_DIR \
            --use_paged_context_fmha enable \
            --max_batch_size 128
        cp $MODEL_SRC/tokenizer* $ENGINE_DIR
        if [ $MODEL_TYPE = "llama" ]; then
            cp model_configs/llama31/* $ENGINE_DIR
        fi
        ;;

    *)
        set +x
        echo "Usage: $0 {all|clean|convert|build}"
        exit 1
        ;;
esac
