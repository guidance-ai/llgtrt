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

### Converting LoRA weights

TensorRT cannot read LoRA weights directly from Huggingface safetensors format. So we convert weights to a simplified "tensor" format that TensorRT can read.  Note that these weights are simply converted -- unlike the base model they are not compiled into an execution graph.  TensorRT provides the _hf_lora_convert.py_ utility to perform this conversion.  This step should be done within the llgtrt container.

```bash
# Create a directory for LoRA weights
mkdir -p /models/lora/my_finetuning

# Convert weights.  Float32 is used only as an intemediate storage type here; the weights will be converted to bfloat16 when loaded for inference.
python3 /code/TensorRT-LLM/examples/hf_lora_convert.py -i /models/My-Finetuned-Meta-Llama-3.1-8B-Instruct -o /models/lora/my_finetuning --storage-type float32

exit
```

### Launching llgtrt with LoRA

When launching llgtrt you must specify the root directory for your various finetuned models as an additional
parameter.

```bash
PORT=3001 ./docker/run.sh /models/Meta-Llama-3.1-8B-Instruct models/lora
```

## Inference

To load a set of LoRA weights into the TensorRT cache, specify _lora_id_ and _lora_dir_ in your inference request:

``` json
{
    "model": "llama",
    "messages": [
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ],
    "lora_id": 17,
    "lora_dir": "my_finetuning"
}
```

Once a set of LoRA weights is loaded, subsequent requests may done using _lora_id_ alone:

``` json
{
    "model": "llama",
    "messages": [
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ],
    "lora_id": 17
}
```

Finally, you may continue to run base model (non-finetuned) inference requests by omitting both LoRA parameters:

``` json
{
    "model": "llama",
    "messages": [
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ],
}
```

There is no need to restart llgtrt or the TensorRT engine to load a new set of LoRA weights into the cache; weights can be loaded on-the-fly as needed.


LoRA support in llgtrt based on this [nVidia example](https://developer.nvidia.com/blog/tune-and-deploy-lora-llms-with-nvidia-tensorrt-llm/).