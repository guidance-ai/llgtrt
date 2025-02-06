import llgtrt_base


class Plugin(llgtrt_base.PluginBase):
    def process_input(self, messages: list[dict], tools: list[dict]) -> dict:
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False
        )
        assert isinstance(rendered, str)
        tokens = self.tokenizer.encode(rendered, add_special_tokens=False)
        print(tokens)
        return {"prompt": rendered, "tokens": tokens}
