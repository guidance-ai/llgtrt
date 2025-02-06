import llgtrt_base
import torch


class Plugin(llgtrt_base.PluginBase):
    def process_input(self, messages: list[dict], tools: list[dict]) -> dict:
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False
        )
        assert isinstance(rendered, str)
        tokens = self.tokenizer.encode(rendered, add_special_tokens=False)

        # tensor-making placeholder
        tokens_tensor = torch.tensor(tokens, device="cuda").unsqueeze(0)

        # make sure we synchronize torch before returning
        torch.cuda.current_stream().synchronize()

        return {
            "prompt": rendered,
            "tokens": tokens,
            "tokens_tensor": llgtrt_base.wrap_tensor(tokens_tensor),
        }
