# llgtrt (llguidance + TensorRT-LLM)

This project demonstrates how to use
[llguidance library](https://github.com/microsoft/llguidance)
for constrained output with
[NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM),
implementing a server with 
[OpenAI REST API](https://platform.openai.com/docs/api-reference/introduction).

The server supports regular completions and chat endpoints
with JSON with schema enforcement ("Structured Output" in OpenAI docs),
as well as full context-free grammars using [Guidance library](https://github.com/guidance-ai/guidance).

This server is similar in spirit to [TensorRT-LLM OpenAI server example](./TensorRT-LLM/examples/apps/openai_server.py),
but python-free and with support for constrained output.
Similarly to the example above, it **does not** use the NVIDIA Triton Inference Server.

## Requirements

You will need a Linux machine with NVIDIA GPU and Docker set up to use the
nvidia-docker runtime.

## Running

Overview of steps:

- build or pull `llgtrt/llgtrt` docker container
- build a trtllm engine (likely using the container)
- create configuration files
- use the container to run the engine

### Building or Pulling Docker Container

To use a pre-built container, run:

```bash
docker pull llgtrt/llgtrt
```

To build a container use:

```bash
./docker/build.sh
```

The build script will initialize submodules if missing.
It takes about 15 minutes on a GitHub runner, should be typically faster on a local machine.

### Building the TensorRT-LLM Engine

This is following 
[TensorRT-LLM Quick-start](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html),
adjusted for running in the `llgtrt/llgtrt` container.
First, use the `llgtrt/llgtrt` container to run bash.

```bash
./docker/bash.sh --volume /path/to/hf-models:/models
```

The following steps are done inside of the container:

```bash
# convert HF model to a checkpoint
python3 /opt/TensorRT-LLM-examples/llama/convert_checkpoint.py \
    --dtype bfloat16 \
    --model_dir /models/Meta-Llama-3.1-8B-Instruct \
    --output_dir /models/model-ckpt \
    --tp_size 1

# then, run trtllm build
trtllm-build --checkpoint_dir /models/model-ckpt \
    --gemm_plugin bfloat16 \
    --output_dir /models/model-engine \
    --use_paged_context_fmha enable

# clean up ckpt (optional)
rm -rf /models/model-ckpt

# finally, copy tokenizer.json and tokenizer_config.json
cp /models/Meta-Llama-3.1-8B-Instruct/tokenizer.json /models/model-engine
cp /models/Meta-Llama-3.1-8B-Instruct/tokenizer_config.json /models/model-engine

# exit the container
exit
```

Make sure to modify the path to the input model (it needs to contain the 
HF Transformers `config.json` as well as the `.safetensors` files and
`tokenizer.json`).
If you're running on more than one 1 GPU, modify the `--tp_size` argument.

### Create config files

By default, llgtrt will use chat template from `tokenizer_config.json`.
If present, it will also read `tokenizer_config_llgtrt.json` from the same directory
and apply any keys from it to `tokenizer_config.json`.

You can also modify TensortRT-LLM's runtime configuration with `runtime.json` file
and `llguidance_parser` configuration with `llguidance.json`.
TODO add more docs

### Running the Engine

```bash
PORT=3001 ./docker/run.sh /path/to/hf-models/model-engine
```

The command will print out the actual `docker run` invocation on first line
if you want to invoke it directly later.
`PORT` defaults to 3000.

You can pass additional arguments after the engine path.
Try running `./docker/run.sh /path/to/hf-models/model-engine --help` for more info.
The `--help` has up-to-date info on `chat.json` and `runtime.json` files -
the options can be specified either in these files (replace `-` with `_`)
or on command line.

## Development

First build the Docker container to be used in the dev container.
If you had already followed steps above, you can skip this.
Otherwise, run `./docker/build.sh`

Next, in VSCode re-open the folder in container.

## Credits

The basic structure of the server borrows inspiration from
[npuichigo/openai_trtllm](https://github.com/npuichigo/openai_trtllm),
which has similar aims, but uses NVidia Triton Server wrapping TensorRT-LLM.

## TODO

- [x] constrained output currently requires n=1
- [x] stop sequence support
- [x] don't use this by default: Capacity Scheduler Policy: GUARANTEED_NO_EVICT
- [x] add script for mpirun auto-detecting engine size
- [ ] multi-LoRA?
- [x] text template for JSON schema (Please follow this schema: ...)
- [x] test with TP=4
- [ ] test phi-3.5
- [ ] multi-modal input
- [ ] when streaming, and stop is set, we need to buffer the output so as not to return the stop sequence itself
- [x] unwind panic for mask computation etc
- [ ] logprobs
- [ ] logprobs with argmax sampling and constraints
- [ ] expose the 'surprise' measure somehow
- [x] for tools, right now it forces a tool - we want to allow either message or tool
- [ ] we use `<|python_tag|>` which is llama 3.1 specific; make it configurable