import requests
# import llgtrt_base
# from llgtrt.py import llgtrt_base
import torch
import numpy as np
import tensorrt_llm
import subprocess

# make sure qwen_vl_utils is installed
# subprocess.call(["pip", "install", "qwen_vl_utils"])

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tensorrt_llm.functional import RotaryScalingType, RopeEmbeddingUtils
from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm.layers.attention import MropeParams
# from llgtrt_native import PluginInit

class PluginInit:
    # based on general llgtrt configuration
    engine_dir: str
    tokenizer_dir: str
    chat_template: str
    bos_token: str | None
    eos_token: str

    # multi-modal configuration, under "py": { ... } in llgtrt.json5
    visual_engine_dir: str
    hf_model_dir: str
    arguments: dict

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

class Plugin:
    def __init__(self, init: PluginInit):
        # super().__init__(init)
        # tokenizer is already initialized by the base class
        # self.tokenizer = AutoTokenizer.from_pretrained(init.hf_model_dir)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            init.hf_model_dir, torch_dtype="float16", device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(init.hf_model_dir)
        # move visual model to GPU - will be replaced by trt visual executor
        self.model.visual.to("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_device = self.model.visual.device

        print("Plugin initialized from HF model directory:", init.hf_model_dir)

    def process_input(
        self, messages: list[dict], tools: list[dict]
    ) -> ProcessInputResult:
        print("process_input called")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image" : "https://lanytek.com/images/demo_lower.png"
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if "pixel_values" not in inputs:
            return ProcessInputResult(
            prompt=text,
            tokens=inputs["input_ids"][0].numpy().tolist()
        )

        # calculate visual-features
        input_ids = inputs["input_ids"].to(self.model.model.device)
        text_embeds = self.model.model.embed_tokens(input_ids).to(self.visual_device)
        image_embeds = None
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"].to(self.visual_device)
            image_grid_thw = inputs["image_grid_thw"].to(self.visual_device)
            pixel_values = pixel_values.type(self.model.visual.get_dtype())
            image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)

            n_image_tokens = (input_ids == self.model.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(text_embeds)
                .to(text_embeds.device)
            )
            image_embeds = image_embeds.to(text_embeds.device, text_embeds.dtype)
            text_embeds = text_embeds.masked_scatter(image_mask, image_embeds)

        position_ids = None
        cache_position = None
        video_grid_thw = None
        rope_deltas = self.model.rope_deltas
        attention_mask = inputs["attention_mask"]

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

        mrope_rotary_cos_sin, mrope_position_deltas = self._compute_mrope_args(position_ids, rope_deltas)

        # replace visual/video pad token with unique token id
        batch_input_ids_list = inputs["input_ids"].cpu().numpy()
        for i in range(batch_input_ids_list.shape[0]):
            input_ids = batch_input_ids_list[i]
            mask = (input_ids == self.model.config.image_token_id) | (
                        input_ids == self.model.config.vision_token_id) | (input_ids == self.model.config.video_token_id)
            indices = np.nonzero(mask)[0]
            value = self.model.config.vocab_size
            for idx in indices:
                input_ids[idx] = value
                value += 1
        input_ids = batch_input_ids_list[0].tolist()
        
        # r = ProcessInputResult(
        #     prompt=text,
        #     tokens=input_ids,
        #     # prompt_table=image_embeds.to("cpu") if image_embeds is not None else None,
        #     # prompt_tasks=[0] if image_embeds is not None else None,
        #     # mrope_rotary_sin_cos=mrope_rotary_cos_sin[0],
        #     # mrope_position_deltas=mrope_position_deltas[0][0].to("cpu").item(),
        #     # skip_cross_attn_blocks=None,
        #     # encoder_input_features=None,
        #     # encoder_output_length=None,
        #     # cross_attention_masks=None,
        #     # input_position_ids=None
        # )

        # r.prompt_table = image_embeds
        # # r.prompt_tasks = [0]
        # r.mrope_rotary_sin_cos = mrope_rotary_cos_sin[0].to("cuda")
        # r.mrope_position_deltas = mrope_position_deltas[0][0].item()

        r = {
            "prompt" : text,
            "tokens" : input_ids,
            "prompt_table": image_embeds.to("cpu"),
            "mrope_rotary_sin_cos": mrope_rotary_cos_sin[0].to("cpu"),
            "mrope_position_deltas": mrope_position_deltas[0][0].item()
        }

        return r

    def _compute_mrope_args(self, position_ids, rope_deltas):
        mrope_position_ids = position_ids
        mrope_position_deltas = rope_deltas
        mrope_position_ids = mrope_position_ids.transpose(1, 0)
        max_position_embeddings = int(self.model.config.max_position_embeddings)
        rotary_embedding_dim = int(self.model.config.hidden_size / self.model.config.num_attention_heads)
        mrope_position_ids_padding = torch.zeros(mrope_position_ids.shape[:-1] +
                                                    (max_position_embeddings, ),
                                                    dtype=torch.int32)
        mrope_position_ids_padding[:, :, :mrope_position_ids.
                                    shape[-1]] = mrope_position_ids

        rotary_embedding_base = float(self.model.config.rope_theta)
        rotary_embedding_scale = float(1.0)
        rotary_embedding_scale_type = RotaryScalingType.mrope
        rotary_embedding_scaling = None
        inv_freq, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            max_position_embeddings, rotary_embedding_dim,
            rotary_embedding_base, rotary_embedding_scale,
            rotary_embedding_scale_type, rotary_embedding_scaling)
        rotary_cos_sin = rotary_cos_sin.reshape(max_position_embeddings,
                                                int(rotary_embedding_dim / 2),
                                                2)
        rotary_cos_sin = torch.from_numpy(rotary_cos_sin)
        cos_ori = rotary_cos_sin[:, :, 0]
        sin_ori = rotary_cos_sin[:, :, 1]
        cos = cos_ori[mrope_position_ids_padding]
        sin = sin_ori[mrope_position_ids_padding]

        mrope_section = [16, 24, 24]
        unsqueeze_dim = -1
        cos = torch.cat([
            m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(unsqueeze_dim)
        sin = torch.cat([
            m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))
        ],
                        dim=-1).unsqueeze(unsqueeze_dim)
        concat_cos_sin = np.concatenate((cos, sin), axis=-1)
        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)
        concat_cos_sin = torch.from_numpy(concat_cos_sin)

        return [concat_cos_sin, mrope_position_deltas]