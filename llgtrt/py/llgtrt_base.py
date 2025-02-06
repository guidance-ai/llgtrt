import llgtrt_native
import transformers
import json
import torch

from typing import Any

WrappedTensor = tuple[torch.Tensor, int, int, tuple[int, ...]]


def _wrap(t: torch.Tensor) -> WrappedTensor:
    tp = llgtrt_native.torch_dtype(str(t.dtype))
    return (t, tp, t.data_ptr(), tuple(t.shape))


def wrap_tensor(t: torch.Tensor) -> WrappedTensor:
    assert t.is_contiguous()
    assert t.device.type == "cuda"
    return _wrap(t)


def wrap_buffer(elts: list[int], dtype: torch.dtype) -> Any:
    t = torch.tensor(elts, dtype=dtype, device="cpu")
    return _wrap(t)


def is_wrapped_tensor(t: WrappedTensor) -> bool:
    return isinstance(t, tuple) and len(t) == 4 and isinstance(t[0], torch.Tensor)


class ProcessInputResult:
    def __init__(self, prompt: str, tokens: list[int]):
        self.prompt = prompt
        self.tokens = tokens
        self.has_tensors = False

        # PromptTuningConfig: embedding_table and input_token_extra_ids
        self.prompt_table: torch.Tensor | None = None
        self.prompt_tasks: list[int] | None = None

        # MropeConfig
        self.mrope_rotary_sin_cos: torch.Tensor | None = None
        self.mrope_position_deltas: int | None = None

        self.skip_cross_attn_blocks: torch.Tensor | None = None

        self.encoder_input_features: torch.Tensor | None = None
        self.encoder_output_length: int | None = None
        self.cross_attention_masks: torch.Tensor | None = None

        self.input_position_ids: list[int] | None = None


class PluginBase:
    def __init__(self, init: llgtrt_native.PluginInit):
        print("Creating tokenizer from", init.tokenizer_folder)
        self.tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file=init.tokenizer_folder + "/tokenizer.json",
            clean_up_tokenization_spaces=False,  # ???
        )
        self.tokenizer.chat_template = init.chat_template
        toks = self.tokenizer.encode("Hello world")
        assert len(toks) > 0
        print("Plugin created")

    def _process_input(self, chat_params: str) -> dict:
        ch = json.loads(chat_params)
        if not ch["tools"]:
            ch["tools"] = None
        r = self.process_input(messages=ch["messages"], tools=ch["tools"])
        assert isinstance(r, ProcessInputResult)

        if r.prompt_tasks is not None:
            r.prompt_tasks = wrap_buffer(r.prompt_tasks, torch.int64)
        if r.input_position_ids is not None:
            r.input_position_ids = wrap_buffer(r.input_position_ids, torch.int64)

        # wrap tensors if any
        for k, v in r.__dict__.items():
            if is_wrapped_tensor(v):
                r.has_tensors = True
            elif isinstance(v, torch.Tensor):
                r.__dict__[k] = wrap_tensor(v)
                r.has_tensors = True

        if r.has_tensors:
            # make sure we synchronize torch before returning
            torch.cuda.current_stream().synchronize()

        return r.__dict__.copy()

    def process_input(
        self, messages: list[dict], tools: list[dict]
    ) -> ProcessInputResult:
        raise NotImplementedError("process_input not implemented")
