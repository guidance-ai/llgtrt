import copy
import requests
import llgtrt_base
import torch

from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from llgtrt_native import PluginInit

class Plugin(llgtrt_base.PluginBase):
    def __init__(self, init: PluginInit):
        super().__init__(init)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            init.hf_model_dir,
            device_map="cpu",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            init.hf_model_dir,
            trust_remote_code=True,
        )

        # move visual model to gpu
        self.model.vision_model = self.model.vision_model.to("cpu")
        self.model.multi_modal_projector = self.model.multi_modal_projector.to("cpu")
        print("Plugin initialized from HF model directory:", init.hf_model_dir)

    def process_input(
        self, params: llgtrt_base.ProcessInputParams
    ) -> llgtrt_base.ProcessInputResult:
        messages = params.messages
        print("process_input called, ", messages)

        messages, urls = self._process_messages(messages)

        images = [Image.open(requests.get(url, stream=True).raw) for url in urls]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if len(images) == 0:
            return llgtrt_base.ProcessInputResult(
            prompt=prompt,
            tokens=self.processor.tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True
            )
        )

        inputs = self.processor(
            images,
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.vision_model.device)

        vision_outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            aspect_ratio_ids=inputs["aspect_ratio_ids"],
            aspect_ratio_mask=inputs["aspect_ratio_mask"],
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        cross_attention_states = vision_outputs[0]
        cross_attention_states = self.model.multi_modal_projector(cross_attention_states).reshape(
            -1, self.model.hidden_size
        )

        cross_attention_mask = _prepare_cross_attention_mask(
            inputs["cross_attention_mask"][0],
            num_vision_tokens=self.model.vision_model.num_patches,
            dtype=self.model.dtype,
            max_new_tokens=params.max_new_tokens
        )

        cross_attention_mask = cross_attention_mask.reshape(-1, cross_attention_states.shape[0])


        r = llgtrt_base.ProcessInputResult(
            prompt=prompt,
            tokens=inputs["input_ids"].cpu().numpy()[0].tolist()
        )
        r.encoder_input_features = cross_attention_states.cuda().half() # Change this to bfloat16 if engine is using bfloat16
        r.cross_attention_masks = (cross_attention_mask).cuda()
        r.skip_cross_attn_blocks = torch.Tensor([False]).cuda()
        r.encoder_output_length = cross_attention_states.shape[0]

        return r
    
    def _process_messages(self, messages: list[dict]):
        urls = []
        messages = copy.deepcopy(messages)
        for m in messages:
            c = m.get("content", None)
            if isinstance(c, list):
                parts_to_change = []
                for part in c:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        urls.append(url)
                        parts_to_change.append(part)

                for part in parts_to_change:
                    part["type"] = "image"
                    part.pop("image_url", None)

        return messages, urls


def _prepare_cross_attention_mask(
    cross_attention_mask: torch.Tensor,
    num_vision_tokens: int,
    dtype: str,
    max_new_tokens=100,
) -> torch.Tensor:
    text_total_length, *_ = cross_attention_mask.shape
    cross_attention_mask = cross_attention_mask.repeat_interleave(
        num_vision_tokens, dim=2)

    cross_attention_mask = cross_attention_mask.view(
        text_total_length, -1)
    cross_attention_mask = cross_attention_mask.unsqueeze(1)
    cross_attention_mask = cross_attention_mask.to(
        dtype).to(torch.bool).reshape(
            [-1, cross_attention_mask.shape[-1]])

    # prepare cross_attention_mask for generation phase and concat them
    tmp_mask = [cross_attention_mask] + [
        cross_attention_mask[-1:, :] for _ in range(max_new_tokens)
    ]
    cross_attention_mask = torch.concat(tmp_mask)

    return cross_attention_mask    