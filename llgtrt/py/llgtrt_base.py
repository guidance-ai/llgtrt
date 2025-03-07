import llgtrt_native
import transformers
import json
import torch

from typing import Any
from dataclasses import dataclass

WrappedTensor = tuple[torch.Tensor, int, int, tuple[int, ...]]


def _wrap(t: torch.Tensor) -> WrappedTensor:
    tp = llgtrt_native.torch_dtype(str(t.dtype))
    return (t, tp, t.data_ptr(), tuple(t.shape))


def wrap_tensor(t: torch.Tensor) -> WrappedTensor:
    assert t.is_contiguous()
    # assert t.device.type == "cuda"
    return _wrap(t)


def wrap_buffer(elts: list[int], dtype: torch.dtype) -> Any:
    t = torch.tensor(elts, dtype=dtype, device="cpu")
    return _wrap(t)


def is_wrapped_tensor(t: WrappedTensor) -> bool:
    return isinstance(t, tuple) and len(t) == 4 and isinstance(t[0], torch.Tensor)


def to_dict(init):
    r = {}
    for k in dir(init):
        if k.startswith("_"):
            continue
        r[k] = getattr(init, k)
    return r


@dataclass
class ProcessInputParams:
    messages: list[dict]
    tools: list[dict] | None
    json_schema: dict | None
    streaming: bool
    max_new_tokens: int
    num_return_sequences: int
    eos_token_id: int
    seed: int

    _chat_params: dict
    _req_params: dict


class ProcessInputResult:
    def __init__(self, prompt: str, tokens: list[int]):
        self.prompt = prompt
        self.tokens = tokens
        self.has_tensors = False

        # PromptTuningConfig: embedding_table and input_token_extra_ids
        self.prompt_table: torch.Tensor | None = None
        self.input_token_extra_ids: list[int] | None = None

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
        d = to_dict(init)
        d["chat_template"] = "..."
        print("PluginInit: ", json.dumps(d, indent=2))

        print("Creating tokenizer from", init.tokenizer_dir)
        self.tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file=init.tokenizer_dir + "/tokenizer.json",
            clean_up_tokenization_spaces=False,  # ???
        )
        self.tokenizer.chat_template = init.chat_template
        self.tokenizer.bos_token = init.bos_token
        self.tokenizer.eos_token = init.eos_token
        toks = self.tokenizer.encode("Hello world")
        assert len(toks) > 0
        print("Base plugin created")

    def _process_input(self, chat_params: str, req_params: str) -> dict:
        chat_p = json.loads(chat_params)
        if not chat_p["tools"]:
            chat_p["tools"] = None
        if not chat_p["json_schema"]:
            chat_p["json_schema"] = None
        req_p = json.loads(req_params)
        process_input_params = ProcessInputParams(
            messages=chat_p["messages"],
            tools=chat_p["tools"],
            json_schema=chat_p["json_schema"],
            streaming=req_p["streaming"],
            max_new_tokens=req_p["max_new_tokens"],
            num_return_sequences=req_p["num_return_sequences"],
            eos_token_id=req_p["eos_token_id"],
            seed=req_p["seed"],
            _chat_params=chat_p,
            _req_params=req_p,
        )

        r = self.process_input(process_input_params)
        assert isinstance(r, ProcessInputResult)

        if r.input_token_extra_ids is not None:
            r.input_token_extra_ids = wrap_buffer(r.input_token_extra_ids, torch.int64)
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

        rr = r.__dict__.copy()
        # print(rr)
        return rr

    def process_input(self, params: ProcessInputParams) -> ProcessInputResult:
        raise NotImplementedError("process_input not implemented")
