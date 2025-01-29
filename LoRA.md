# LoRA support in llgtrt

TensorRT provides full support for parameter-efficient finetuning via a technicque called _Low Rank Adaptation_, commonly abbreviated as _LoRA_.  The engine even provides support for multiple simultaneous sets of finetuned weights (often known as _Multi-LoRA_), though configuring this requires a few extra steps, both during setup and inference.

The following extra steps are required during setup:

1. Compile the base model weights with the LoRA module activated.
2. Convert each set of LoRA weights from standard Huggingface format to a specialized tensor format compatible with TensorRT.  Store these in a directory accessible from your TensorRT installation.
3. Launch llgtrt/TensorRT with an extra parameter pointing to your LoRA weights.

During inference, TensorRT maintains a cache of LoRA weights within GPU memory where the precise number of LoRA models that can fit in the cache varies based on your configuration.  To perform inference using a specific LoRA model you must first load its weights into the cache keyed by a unique integer identifier.  Once weights are loaded then subsequent inference requests may be made using the integer Id alone.  Depending on the size of your cache, loading a new set of LoRA weights may result in another set being evicted from the cache.

## Setup details

Detailed instructions on how to prepare a llgtrt/TensorRT instance for LoRA:

### Compiling the base model

From within the llgtrt Docker container, use _convert_checkpoint.py_ as you normally would but then add additional LoRA-specific parameters when calling _trtllm-build_:

```bash
# Convert the HF model to a checkpoint
python3 /opt/TensorRT-LLM-examples/llama/convert_checkpoint.py \
    --dtype bfloat16 \
    --model_dir /models/Meta-Llama-3.1-8B-Instruct \
    --output_dir /models/model-ckpt \
    --tp_size 1

# Run trtllm build with LoRA enabled.  For this example we activate the attn_q, attn_k, and attn_v modules.
trtllm-build --checkpoint_dir /models/model-ckpt \
    --gemm_plugin bfloat16 \
    --output_dir /models/model-engine \
    --use_paged_context_fmha enable \
    --lora_plugin bfloat16 \
    --lora_dir /models/My-Finetuned-Meta-Llama-3.1-8B-Instruct \
    --lora_ckpt_source hf \
    --lora_target_modules attn_q attn_k attn_v

# Clean up checkpoint (optional)
rm -rf /models/model-ckpt

# Finally, copy tokenizer.json and tokenizer_config.json
cp /models/Meta-Llama-3.1-8B-Instruct/tokenizer.json /models/model-engine
cp /models/Meta-Llama-3.1-8B-Instruct/tokenizer_config.json /models/model-engine

# Exit the container
exit
```

### Extract LoRA weights

TensorRT cannot read LoRA weights directly from Huggingface safetensors format. So we provide a Python-based utility to extract weights into a simplified "tensor" format that TensorRT can read.
Note that these weights are simply converted -- unlike the base model they are not compiled into an execution graph.
The scripts/extrct_lora.py utility performs the extraction.  It should be run within the llgtrt container.

```bash
# Ensure that a base LoRA directory exists
mkdir -p /models/lora

# Run the extraction
python3 /code/scripts/extract_lora.py -i /models/My-Finetuned-Meta-Llama-3.1-8B-Instruct -o /models/lora/my_finetuning.safetensors

exit
```
The datatype of the LoRA weights must be the same as the datatype of the base model.  In the event that these differ, you can optionally
add a --dtype parameter (short form -t) to force a specific output type.  For example:

```bash
python3 /code/scripts/extract_lora.py -i /models/My-Finetuned-Meta-Llama-3.1-8B-Instruct -o /models/lora/my_finetuning.safetensors -t bfloat16
```


### Launching llgtrt with LoRA

When launching llgtrt you must specify the root directory for your various finetuned models as an additional
parameter.

```bash
PORT=3001 ./docker/run.sh /models/Meta-Llama-3.1-8B-Instruct models/lora
```

## Inference

To load a set of LoRA weights into the TensorRT cache, specify _lora_model_ in your inference request:

``` json
{
    "model": "llama",
    "messages": [
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ],
    "lora_model": "my_finetuning"
}
```

TensorRT maintains a cache of LoRA weights within GPU memory. The first time you use a given lora_model it will be loaded into the cache.
Subsequent requests using the same lora_model will use the cache.  In the event the cache fills up, TensorRT will evict a set of LoRA
weights from the cache (its choice) and the next request for that lora_model will reload its weights into the cache.

For finer control over the cache we provide the *load_lora_weights* parameter.  Setting this to "always" will always load weights
even if they're already cached.  Likewise, setting this to "never" will never load weights, producing an error in the event the
weights are not cached.  For example:

``` json
{
    "model": "llama",
    "messages": [
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ],
    "lora_model": "my_finetuning",
    "load_lora_weights": "never"
}
```


There is no need to restart llgtrt or the TensorRT engine to load a new set of LoRA weights into the cache; weights can be loaded on-the-fly as needed.


LoRA support in llgtrt based on this [nVidia example](https://developer.nvidia.com/blog/tune-and-deploy-lora-llms-with-nvidia-tensorrt-llm/).