import llgtrt_native
import transformers
import json


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
        return self.process_input(messages=ch["messages"], tools=ch["tools"])

    def process_input(self, messages: list[dict], tools: list[dict]) -> dict:
        raise NotImplementedError("process_input not implemented")
