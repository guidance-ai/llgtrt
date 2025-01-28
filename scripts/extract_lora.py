#! /usr/bin/env python3
import argparse
import datetime
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import safetensors

from tensorrt_llm.lora_manager import LoraManager
from tensorrt_llm.models.convert_utils import get_model_path, load_state_dict
from tensorrt_llm._utils import str_dtype_to_torch

log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
logging.basicConfig(format=log_format)
LOGGER = logging.getLogger(__name__)

def get_all_lora_weights(lora_weights):
    all_weights = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(
        r'.*\.layers\.([0-9]+)\.(self_attn|mlp)\.([a-z_]+)\.lora_(A|B)\.weight.*'
    )
    moe_pattern = re.compile(
        r'.*\.layers\.([0-9]+)\.(block_sparse_moe)\.((experts)\.([0-9]+)\.|)([a-zA-Z0-9_]+)\.lora_(A|B)\.weight.*'
    )
    for key, weights in lora_weights.items():
        m = pattern.match(key)
        m_moe = moe_pattern.match(key)
        if m:
            layer_idx = int(m.group(1))
            hf_module = m.group(3)
            inout = "in" if m.group(4) == "A" else "out"
            all_weights[layer_idx][hf_module][inout] = weights
        elif m_moe:
            layer_idx = int(m_moe.group(1))
            hf_module = m_moe.group(6)
            inout = "in" if m_moe.group(7) == "A" else "out"
            all_weights[layer_idx][hf_module][inout] = weights
        else:
            print(f"no match {key}")
            continue
    return all_weights


def preprocess_lora_weights(lora_model):
    # Swap weights of gate_up_proj
    for key, value in lora_model.items():
        if "gate_up_proj.lora_B.weight" in key:
            print("Swap {}".format(key))
            original_weights = value.contiguous().clone()
            half_split = original_weights.shape[0] // 2
            first_half = original_weights[:half_split, :]
            second_half = original_weights[half_split:, :]
            value = torch.cat((second_half, first_half), dim=0)
            lora_model[key] = value
    return lora_model


hf_modules_to_trtllm_modules = {
    "q_proj": "attn_q",
    "v_proj": "attn_v",
    "k_proj": "attn_k",
    "qkv_proj": "attn_qkv",
    "query_key_value": "attn_qkv",
    "o_proj": "attn_dense",
    "dense": "attn_dense",
    "gate_proj": "mlp_h_to_4h",
    "down_proj": "mlp_4h_to_h",
    "up_proj": "mlp_gate",
    "gate_up_proj": "mlp_h_to_4h",
    "c_fc": "mlp_h_to_4h",
    "c_proj": "mlp_4h_to_h",
    "w1": "moe_h_to_4h",
    "w2": "moe_4h_to_h",
    "w3": "moe_gate",
    "gate": "moe_router",
}  # lora modules on llama
hf_modules_to_module_id = {
    k: LoraManager.LORA_MODULE_IDS[v]
    for k, v in hf_modules_to_trtllm_modules.items()
}


def convert_hf_model(model_dir, out_file, dtype=None):
    with open(f"{model_dir}/adapter_config.json", "r") as f:
        config = json.load(f)

    alpha = config.get("lora_alpha")
    use_rslora = config.get("use_rslora", False)

    lora_model = load_state_dict(get_model_path(model_dir, "adapter_model"))
    lora_model = preprocess_lora_weights(lora_model)
    all_weights = get_all_lora_weights(lora_model)
    converted_weights = []
    converted_config = []
    for layer_idx, layer_weights in all_weights.items():
        for hf_module, module_weights in layer_weights.items():
            in_weights = module_weights['in']
            out_weights = module_weights['out']
            in_out_weights = []
            adapter_size = 0
            for w, inout in ((in_weights, "in"), (out_weights, "out")):
                assert len(w.shape) == 2
                # assume that the hidden dim is the larger of the 2
                dim0 = w.shape[0]
                dim1 = w.shape[1]
                adapter_size = min(dim0, dim1)
                # in_weights should have shape [adaper_size, hidden]
                if dim1 < dim0 and inout == "in":
                    adapter_size = dim1
                    w = w.transpose(1, 0)
                # out_weights should have shape [hidden, adapter_size]
                elif dim0 < dim1 and inout == "out":
                    adapter_size = dim0
                    w = w.transpose(1, 0)
                if inout == "out":
                    if use_rslora:
                        scale = alpha / np.sqrt(adapter_size)
                    else:
                        scale = alpha / adapter_size
                    w = w * scale
                w = w.contiguous().flatten()
                in_out_weights.append(w)
            in_out_weights = torch.concatenate(in_out_weights).flatten()
            converted_weights.append(in_out_weights)
            converted_config.append(
                [hf_modules_to_module_id[hf_module], layer_idx, adapter_size])
    max_row_size = 0
    for t in converted_weights:
        max_row_size = max(max_row_size, t.shape[0])
    for i in range(len(converted_weights)):
        converted_weights[i] = torch.nn.functional.pad(
            converted_weights[i],
            (0, max_row_size - converted_weights[i].shape[0])).unsqueeze(0)
    converted_weights = torch.concatenate(
            converted_weights,
            dim=0)
    if (dtype is not None):
        converted_weights = converted_weights.to(dtype=str_dtype_to_torch(dtype))
    converted_weights = converted_weights.cpu()
    
    converted_config = torch.tensor(converted_config,
                                    dtype=torch.int32,
                                    device='cpu')
    output_dict = {
        'weights': converted_weights,
        'config': converted_config
    }

    safetensors.torch.save_file(output_dict, out_file)


def main(args):
    start_time = datetime.datetime.now()
    convert_hf_model(args.in_file, args.out_file, args.dtype)

    LOGGER.info("Spent %s (h:m:s) to convert the prompt model",
                datetime.datetime.now() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-file',
        '-o',
        type=Path,
        help='path to output extracted LoRA weights and config in safetensors format',
        required=True)
    parser.add_argument('--in-file',
                        '-i',
                        type=Path,
                        help='path to input lora checkpoint file',
                        required=True)
    parser.add_argument('--dtype',
                        '-t',
                        help='datatype to convert extracted LoRA weights to')
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    args = parser.parse_args()

    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    main(args)
