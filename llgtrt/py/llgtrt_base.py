import llgtrt_native
import transformers
import json

class PluginBase:
    def __init__(self, init: llgtrt_native.PluginInit):
        print("Creating tokenizer from", init.tokenizer_folder)
        self.tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file=init.tokenizer_folder + "/tokenizer.json",
            clean_up_tokenization_spaces=False, # ???
        )
        toks = self.tokenizer.encode("Hello world")
        assert len(toks) > 0
        print("Plugin created")

    def _process_input(self, messages: str) -> dict:
        return self.process_input(json.loads(messages))

    def process_input(self, messages: list[dict]) -> dict:
        raise NotImplementedError("process_input not implemented")