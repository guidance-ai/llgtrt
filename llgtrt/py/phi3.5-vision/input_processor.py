import copy
import requests
import llgtrt_base
import torch

import numpy as np
import tensorrt_llm
import subprocess

subprocess.call(["pip", "install", "flash-attn", "--no-build-isolation"])

from torch import nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from llgtrt_native import PluginInit

class Plugin(llgtrt_base.PluginBase):
    def __init__(self, init: PluginInit):
        super().__init__(init)
        self.model = AutoModelForCausalLM.from_pretrained(
            init.hf_model_dir, 
            device_map="cpu", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            # _attn_implementation='flash_attention_2'
        )
        self.processor = AutoProcessor.from_pretrained(
            init.hf_model_dir, 
            trust_remote_code=True, 
            num_crops=4
        )

        # move visual model to gpu
        self.model.model.vision_embed_tokens.img_processor = self.model.model.vision_embed_tokens.img_processor.to("cuda")
        self.model.model.vision_embed_tokens.img_projection = self.model.model.vision_embed_tokens.img_projection.to("cuda")
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

        inputs = self.processor(prompt, images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")
        num_images, num_crops, c, h, w = pixel_values.shape
        assert c == 3 and h == w == 336

        img_features = self.model.model.vision_embed_tokens.get_img_features(
            pixel_values.flatten(0, 1)).reshape(
            num_images, num_crops, -1, self.model.model.vision_embed_tokens.image_dim_out
        ).to("cuda")

        image_sizes = inputs["image_sizes"].to("cuda")
        image_features_proj = hd_feature_transform(self.model.model.vision_embed_tokens, img_features, image_sizes)

        # process input_ids
        input_ids = inputs["input_ids"].cpu().numpy()[0].tolist()
        v = self.model.config.vocab_size
        for i in range(len(input_ids)):
            if input_ids[i] == -1:
                input_ids[i] = v
                v += 1

        r = llgtrt_base.ProcessInputResult(
            prompt=prompt,
            tokens=input_ids
        )
        r.prompt_table = image_features_proj

        return r

    def _process_messages(self, messages: list[dict]):
        urls = []
        messages = copy.deepcopy(messages)
        for m in messages:
            c = m.get("content", None)
            if isinstance(c, list):
                placeholder = ""
                text_content = ""
                for part in c:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        urls.append(url)
                        placeholder += f"<|image_{len(urls)}|>\n"
                    elif part["type"] == "text":
                        text_content = part["text"]

                if placeholder != "":
                    m["content"] = placeholder + text_content

        return messages, urls

# Modified from phi3 code to optimize for running only vision model on GPU
def hd_feature_transform(vision_embed_tokens, image_features, image_sizes):
    """
    image_features: (num_images, num_crops+1, 24*24, 1024)
    """
    assert (
        vision_embed_tokens.hd_transform_order == 'sub_glb'
    ), f'hd_transform_order `{vision_embed_tokens.hd_transform_order}` not implemented'
    if isinstance(vision_embed_tokens.img_projection, nn.Sequential):
        target_device = vision_embed_tokens.img_projection[0].bias.device
        target_dtype = vision_embed_tokens.img_projection[0].bias.dtype
    else:  # It's a single nn.Linear layer
        target_device = vision_embed_tokens.img_projection.bias.device
        target_dtype = vision_embed_tokens.img_projection.bias.dtype

    global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
    # global feature can be viewed as a special HD case with num_crops 1x1
    global_image_features_hd = vision_embed_tokens.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
    #global_image_features_hd_newline = vision_embed_tokens.add_image_newline(global_image_features_hd)
    global_image_features_hd_newline = add_image_newline(vision_embed_tokens, global_image_features_hd)

    all_image_embeddings = []
    # need a for loop to process each image because of different image sizes
    # (patch arrangement is different for each image)
    for i, img_size in enumerate(image_sizes):
        h, w = img_size
        h_crop = h // 336
        w_crop = w // 336
        num_crops = h_crop * w_crop

        # NOTE: real num_crops is padded
        # (num_crops, 24*24, 1024)
        sub_image_features = image_features[i, 1 : 1 + num_crops]
        sub_image_features_hd = vision_embed_tokens.reshape_hd_patches_2x2merge(
            sub_image_features, h_crop, w_crop
        )
        sub_image_features_hd_newline = add_image_newline(vision_embed_tokens, sub_image_features_hd)

        # [sub features, separator, global features]
        all_image_embeddings.extend(
            [
                sub_image_features_hd_newline.squeeze(0).to(target_device),  # (h_crop*12*(w_crop*12+1), 4096)
                vision_embed_tokens.glb_GN.squeeze(0).to(target_device),
                global_image_features_hd_newline[i].to(target_device),
            ]
        )

    image_features_proj = vision_embed_tokens.img_projection(
        torch.cat(all_image_embeddings, dim=0).to(target_device)
    )

    return image_features_proj

def add_image_newline(vision_embed_tokens, image_features_hd):
    """
    image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
    output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
    """
    num_images, h, w, hid_dim = image_features_hd.shape
    # add the newline token to the HD image feature patches
    newline_embeddings = vision_embed_tokens.sub_GN.expand(num_images, h, -1, -1)  # (n_img, h, 1, hid_dim)
    newline_embeddings = newline_embeddings.to(image_features_hd.device)
    image_features_hd_newline = torch.cat(
        [image_features_hd, newline_embeddings], dim=2
    ).reshape(num_images, -1, hid_dim)
    return image_features_hd_newline
