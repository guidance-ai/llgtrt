# llgtrt (llguidance + TensorRT-LLM)

This project implements a REST HTTP server with 
[OpenAI-compatible API](https://platform.openai.com/docs/api-reference/introduction),
based on [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
and [llguidance library](https://github.com/microsoft/llguidance) for constrained output.

The server supports regular completions and chat endpoints with JSON schema enforcement ("Structured Output"), as well as full context-free grammars using the [Guidance library](https://github.com/guidance-ai/guidance).

This server is similar in spirit to the [TensorRT-LLM OpenAI server example](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/apps/openai_server.py), but it is Python-free (implemented in Rust) and includes support for constrained output. Like the example above, it **does not** use the NVIDIA Triton Inference Server.

## Structured Output

The sampling can be constrained by the [Low-Level Guidance library](https://github.com/microsoft/llguidance), part of the [Guidance project](https://github.com/guidance-ai/guidance). While TensorRT computes logits (token probabilities) for the next token, the llguidance library computes a set of tokens allowed by the grammar (whether JSON schema, regular expression, or a full context-free grammar (CFG)) in the form of a bitmask. When both the logits and bitmask are ready, a custom CUDA kernel applies the mask to the logits, and the result is used for sampling inside of TensorRT-LLM.

There is no significant startup cost for all realistic sizes of grammars (no measurable impact on time to first token (TTFT)). The overhead on generation speed (median time between tokens (TBT)) is typically 1-3% (and comes mostly from apply masking kernels on the GPU). The mask computation takes on the order of 100 us of single-core CPU time per token per sequence in the batch. Thus, with 16 cores and a TBT of around 10 ms, batch sizes of up to 1600 are not CPU-bound. Typically, the unconstrained TBT is higher at such batch sizes, and more cores are available, so batch size is not a problem in production.

This approach differs from [Outlines](https://github.com/dottxt-ai/outlines) and [XGrammar](https://github.com/mlc-ai/xgrammar) (which both pre-compute masks, resulting in a startup cost and limits on schema complexity) and is more similar in spirit to [llama.cpp grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md), though it is much faster due to the use of a custom lexer with [derivative-based regexes](https://github.com/microsoft/derivre), an Earley parser, and a [highly optimized](https://github.com/guidance-ai/llguidance/blob/main/docs/optimizations.md) token prefix tree.

### Lark grammars

Llgtrt follows OpenAI API when `response_format` is set to `json_object` or `json_schema` 
(including handling of `strict` field in the schema).

For more flexible constraints, `response_format` can be set to `lark_grammar`.
For example, you can `POST` to `/v1/chat/completions`:

```json
{ "model": "model", 
  "messages": [
    { "role": "user",
      "content": "Please tell me a one line joke."
    } ],
  "response_format": {
    "type": "lark_grammar",
    "lark_grammar": "start: /[A-Z ]+/"
  },
  "max_tokens": 100
}
```

This results in a (bad) joke in uppercase.

Another example involves reasoning models distilled from Deepseek-R1
(the chat format in these models seems to already include `<think>\n`,
so it should not be part of the grammar):

```json
{ "model": "model", "messages": [
    { "role": "user",
      "content": "How many 'r' in strawberry?"
    } ],
  "response_format": {
    "type": "lark_grammar",
    "lark_grammar": "start: /(.|\\n){1000,2000}/ </think> \"\\\\boxed{\" /[0-9]+/ \"}\""
  },
  "max_tokens": 1000
}
```

The `"lark_grammar"` is JSON-encoded version of 
`start: /(.|\n){1000,2000}/ </think> "\\boxed{" /[0-9]+/ "}"`.
Of course you can also use `{0,2000}` to only place upper bound on thinking,
`{1000,}` to place lower bound, or `*` to avoid any bounds.

You can [convert GBNF](https://github.com/guidance-ai/llguidance/blob/main/python/llguidance/gbnf_to_lark.py) grammars to Lark syntax, as it's strictly more expressive.
Learn more in [llguidance docs](https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md).


## Requirements

You will need a Linux machine with an NVIDIA GPU and Docker set up to use the `nvidia-docker` runtime.

So far, we have only tested it on 4xA100 (and single A100).

## Running

Overview of steps:

- Build or pull the `llgtrt/llgtrt` Docker container.
- Build a trtllm engine (likely using the container).
- Create configuration files.
- Use the container to run the engine.

### Building or Pulling Docker Container

To use a pre-built container, run:

```bash
docker pull llgtrt/llgtrt
```

To build a container, use:

```bash
./docker/build.sh
```

The build script will initialize submodules if they are missing. It takes about 15 minutes on a GitHub runner and should typically be faster on a local machine.

#### Optional: Building TensorRT-LLM from source

The build process above uses prebuilt binaries from a release version of TensorRT-LLM.  It is also possible to build your own version of TensorRT-LLM from source and create a build of llgtrt based on that.  This can be used to build a version of llgtrt that will work with versions of TensorRT-LLM newer than the released versions in nVidia's repositories.

To do so, first build TensorRT-LLM from source following the instructions in https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html

Now, build llgtrt based on the Docker image you built above
```bash
./docker/build.sh --trtllm tensorrt_llm/release
```

### Building the TensorRT-LLM Engine

This is based on the [TensorRT-LLM Quick-start](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html).
Follow the steps here, and look into that guide if needed.

First, use the `llgtrt/llgtrt` container to run bash.

```bash
./docker/bash.sh --volume /path/to/hf-models:/models
```

The following steps are done inside the container:

```bash
# Convert the HF model to a checkpoint
python3 /opt/TensorRT-LLM-examples/llama/convert_checkpoint.py \
    --dtype bfloat16 \
    --model_dir /models/Meta-Llama-3.1-8B-Instruct \
    --output_dir /models/model-ckpt \
    --tp_size 1

# Then, run trtllm build
trtllm-build --checkpoint_dir /models/model-ckpt \
    --gemm_plugin bfloat16 \
    --output_dir /models/model-engine \
    --use_paged_context_fmha enable

# Clean up checkpoint (optional)
rm -rf /models/model-ckpt

# Finally, copy tokenizer and preprocessor files to engine folder
cp /models/Meta-Llama-3.1-8B-Instruct/tokenizer*.json /models/model-engine
cp /models/Meta-Llama-3.1-8B-Instruct/preprocessor*.json /models/model-engine # this may be missing

# Exit the container
exit
```

Make sure to modify the path to the input model (it needs to contain the HF Transformers `config.json` as well as the `.safetensors` files and `tokenizer.json`). If you're running on more than one GPU, modify the `--tp_size` argument.

### Running the Engine

```bash
PORT=3001 ./docker/run.sh /path/to/hf-models/model-engine
```

The command will print out the actual `docker run` invocation on the first line if you want to invoke it directly later. `PORT` defaults to 3000.

### Update Configuration (Optional)

The defaults should be mostly reasonable, but you can modify them. First, generate a template configuration file:

```bash
./docker/run.sh /path/to/hf-models/model-engine --print-config > llgtrt.json5
```

The file will contain commented-out defaults for all supported options (JSON5 is a superset of JSON, so you can use comments). Edit it and move it to the engine folder.

To modify the chat template, you can either use `--print-complete-config`, which will include the chat template from `tokenizer_config.json`, or preferably create a separate `chat_template.j2` file in the engine folder:

```bash
./docker/run.sh /path/to/hf-models/model-engine --print-chat-template > chat_template.j2
mv chat_template.j2 /path/to/hf-models/model-engine
```

The paths to `llgtrt.json5` and `chat_template.j2` are controlled by command line arguments. See `--help` for more info:

```bash
./docker/run.sh /path/to/hf-models/model-engine --help
```

You can specify multiple JSON5 config files, and they will be merged in the order specified (with later ones overriding earlier ones). This way, you can separate configuration for the tokenizer, runtime, and guidance parser.

### Running phi-3

The phi-3 tokenizer, while based on llama2 is slightly different.
Drop the following `llgtrt.json5` file in engine folder:

```json5
{
  "tokenizer": {
    "bos_token": null,
    "n_vocab_override": 32064
  }
}
```

## LoRA and Multi-LoRA

Support for finetuned models via LoRA is provided but requires a few extra steps.
Detailed instructions are in [LoRA.md](LoRA.md)

## Development

First, build the Docker container to be used in the dev container. If you have already followed the steps above, you can skip this. Otherwise, run `./docker/build.sh`.

Next, in VSCode, reopen the folder in the container.

## Credits

The basic structure of the server borrows inspiration from [npuichigo/openai_trtllm](https://github.com/npuichigo/openai_trtllm), which has similar aims but uses NVIDIA Triton Server wrapping TensorRT-LLM.

## TODO

- [X] multi-LoRA
- [x] test phi-3.5
- [ ] multi-modal input
- [ ] when streaming, and stop is set, we need to buffer the output so as not to return the stop sequence itself
- [ ] logprobs (currently only work with TP>1; TRTLLM bug?)
- [ ] logprobs with argmax sampling and constraints
- [ ] expose the 'surprise' measure somehow
