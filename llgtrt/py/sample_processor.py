import llgtrt_base
from llgtrt_native import PluginInit
import torch
import transformers

class Plugin(llgtrt_base.PluginBase):
    def __init__(self, init: PluginInit):
        super().__init__(init)
        self.processor = transformers.AutoProcessor.from_pretrained(init.hf_model_dir)


    def process_input(
        self, messages: list[dict], tools: list[dict]
    ) -> llgtrt_base.ProcessInputResult:
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False
        )
        assert isinstance(rendered, str)
        tokens = self.tokenizer.encode(rendered, add_special_tokens=False)

        r = llgtrt_base.ProcessInputResult(prompt=rendered, tokens=tokens)

        # testing tensor passing
        r.prompt_table = torch.rand([100, 200], device="cuda", dtype=torch.float32)
        r.prompt_tasks = [0 for _ in tokens]

        return r
