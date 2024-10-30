# llgtrt (llguidance + TensorRT-LLM)

This project demonstrates how to use
[llguidance library](https://github.com/microsoft/llguidance)
for constrained output with
[NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM),
implementing a REST server compatible with
[OpenAI APIs](https://platform.openai.com/docs/api-reference/introduction).

The server supports regular completions and chat endpoints
with JSON with schema enforcement ("Structured Output"),
as well as full context-free grammars using [Guidance library](https://github.com/guidance-ai/guidance).

This server is similar in spirit to [TensorRT-LLM OpenAI server example](./TensorRT-LLM/examples/apps/openai_server.py),
but is Python-free (implemented in Rust) and with support for constrained output.
Similarly to the example above, it **does not** use the NVIDIA Triton Inference Server.

## Structured Output

The sampling can be constrained by the [Low-Level Guidance library](https://github.com/microsoft/llguidance),
part of the [Guidance project](https://github.com/guidance-ai/guidance).
While TensorRT is computing logits (token probabilities) for the next token,
the llguidance library computes a set of tokens allowed by the grammar
(be it JSON schema, regular expression, or a full context-free grammar (CFG)),
in a form of a bitmask.
When both logits and bitmask are ready, a custom CUDA kernel applies the mask
to the logits, and the result is used for sampling inside of TensorRT-LLM.

There is no significant startup cost for all realistic sizes of grammars
(no measured impact on time to first token (TTFT)).
The overhead on generation speed (median time between tokens (TBT)) is typically 1-3%.
The mask computation takes on the order of 1ms single-core CPU time per token per sequence in batch.
Thus, with 16 cores and TBT of around 10ms, batch sizes of up to 160 are not CPU-bound.
Typically, the unconstrained TBT are higher at such batch sizes though,
and more cores are available, so the batch size is not a problem in production.

Note that this is unlike [Outlines](https://github.com/dottxt-ai/outlines)
(which pre-computes masks, with a startup cost and limits on schema complexity),
and similar in spirit to
[llama.cpp grammars](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md),
though much faster due to usage of a custom lexer with
[derivative-based regexes](https://github.com/microsoft/derivre),
an Earley parser, as well as
[highly optimized](https://github.com/microsoft/toktrie/blob/main/implementation.md)
token prefix tree.

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

### Running the Engine

```bash
PORT=3001 ./docker/run.sh /path/to/hf-models/model-engine
```

The command will print out the actual `docker run` invocation on first line
if you want to invoke it directly later.
`PORT` defaults to 3000.

### Update Configuration (optional)

The defaults should be mostly reasonable, but you can modify them.
First, generate a template configuration file:

```bash
./docker/run.sh /path/to/hf-models/model-engine --print-config > llgtrt.json5
```

The file will contain commented out defaults for all supported options
(JSON5 is a superset of JSON, so you can use comments).
Edit it, and move to the engine folder.

To modify the chat template, you can either use `--print-complete-config`
above which will include the chat template from `tokenizer_config.json`,
or preferably create a separate `chat_template.j2` file in the engine folder:

```bash
./docker/run.sh /path/to/hf-models/model-engine --print-chat-template > chat_template.j2
mv chat_template.j2 /path/to/hf-models/model-engine
```

The paths to `llgtrt.json5` and `chat_template.j2` are controlled by command
line arguments, see `--help` for more info:

```bash
./docker/run.sh /path/to/hf-models/model-engine --help
```

You can even specify several JSON5 config files, and they will be merged
in the order they are specified (with later ones overriding the earlier ones).
This way, you can separate configuration for tokenizer, runtime, and guidance parser.

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

- [ ] multi-LoRA?
- [ ] test phi-3.5
- [ ] multi-modal input
- [ ] when streaming, and stop is set, we need to buffer the output so as not to return the stop sequence itself
- [ ] logprobs (currently only work with TP>1; TRTLLM bug?)
- [ ] logprobs with argmax sampling and constraints
- [ ] expose the 'surprise' measure somehow
