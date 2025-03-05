# LLGTRT Preprocess func for Qwen2-VL

## Installation

Extract llm engine from hf-model
(ref: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/multimodal#qwen2-vl)

```
export MODEL_NAME="Qwen2-VL-2B-Instruct"
git clone https://huggingface.co/Qwen/${MODEL_NAME} `pwd`/${MODEL_NAME}
```

```
python3 TensorRT-LLM/examples/qwen/convert_checkpoint.py \
    --model_dir=`pwd`/${MODEL_NAME} \
    --output_dir=`pwd`/trt_models/${MODEL_NAME} \
    --dtype bfloat16
```

```
trtllm-build --checkpoint_dir `pwd`/trt_models/${MODEL_NAME} \
    --output_dir `pwd` \
    --gemm_plugin=bfloat16 \
    --gpt_attention_plugin=bfloat16 \
    --max_batch_size=1 \
    --max_input_len=2048 --max_seq_len=3072 \
    --max_prompt_embedding_table_size=1296
```

```
rm -rf `pwd`/trt_models
```