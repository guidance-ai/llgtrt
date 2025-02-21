import torch
import tensorrt_llm
import numpy as np
from transformers import AutoTokenizer
from typing import Union
from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm.bindings.executor import MropeConfig
from tensorrt_llm.bindings.executor import PromptTuningConfig
from llgtrt.py.standalone_example.input_processor import Plugin

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

np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})
np_float8 = np.dtype('V1', metadata={"dtype": "float8"})

def numpy_to_torch(x):
    if x.dtype == np_bfloat16:
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    elif x.dtype == np_float8:
        return torch.from_numpy(x.view(np.int8)).view(torch.float8_e4m3fn)
    else:
        return torch.from_numpy(x)

def _prepare_embedding_table(prompt_table: Union[str, torch.Tensor]):
        if isinstance(prompt_table, str):
            prompt_table_data = numpy_to_torch(
                np.load(prompt_table)).to(dtype=plugin.model.dtype)
        else:
            assert isinstance(
                prompt_table,
                torch.Tensor), "Prompt table should be str or torch.Tensor"
            prompt_table_data = prompt_table.to(dtype=plugin.model.dtype)

        return prompt_table_data

def prepare_ptuning_executor(batch_input_ids_list, prompt_table,
                                  prompt_tasks, input_token_extra_ids):
    if input_token_extra_ids:
        assert len(batch_input_ids_list) == len(input_token_extra_ids), \
            f"Batch size of input_token_extra_ids ({len(input_token_extra_ids)}) must be the same as input batch size ({len(batch_input_ids_list)})"
    prompt_tuning_configs = len(batch_input_ids_list) * [None]
    if prompt_table is not None:
        prompt_table_data = _prepare_embedding_table(
            prompt_table).cuda()
        if prompt_tasks is not None:
            task_indices = [int(t) for t in prompt_tasks.split(',')]
            assert len(task_indices) == len(batch_input_ids_list), \
                f"Number of supplied tasks ({len(task_indices)}) must match input batch size ({len(batch_input_ids_list)})"
            prompt_tuning_configs = [
                PromptTuningConfig(
                    embedding_table=prompt_table_data[task_indices[i]],
                    input_token_extra_ids=input_token_extra_ids[i]
                    if input_token_extra_ids else None)
                for i in range(len(batch_input_ids_list))
            ]
        else:
            prompt_tuning_configs = [
                PromptTuningConfig(
                    embedding_table=prompt_table_data[0],
                    input_token_extra_ids=input_token_extra_ids[i]
                    if input_token_extra_ids else None)
                for i in range(len(batch_input_ids_list))
            ]
    return prompt_tuning_configs

hf_dir = "/home/jc1da/repos/llgtrt/models/trt_engines/Qwen2-VL-2B-Instruct/fp16/1-gpu/llm/Qwen2-VL-2B-Instruct"
llm_engine_dir = "/home/jc1da/repos/llgtrt/models/trt_engines/Qwen2-VL-2B-Instruct_new/fp16/1-gpu/llm"

plugin_init = PluginInit()
plugin_init.hf_model_dir = hf_dir
plugin = Plugin(plugin_init)

response = plugin.process_input([], [])
plugin_response = response

tokenizer = AutoTokenizer.from_pretrained(hf_dir) 
trt_model = ModelRunnerCpp.from_dir(
                    llm_engine_dir,
                    rank=tensorrt_llm.mpi_rank(),
                    debug_mode=False,
                )

output_config = tensorrt_llm.bindings.executor.OutputConfig()
mrope_config = MropeConfig(mrope_rotary_cos_sin=response["mrope_rotary_sin_cos"], mrope_position_deltas=response["mrope_position_deltas"])
if len(response["prompt_table"].shape) == 2:
    response["prompt_table"] = response["prompt_table"].unsqueeze(0)
#ptuning_object = prepare_ptuning_executor([response["tokens"]], response["prompt_table"], ''.join(map(str,response.prompt_tasks)), None)[0]
ptuning_object = prepare_ptuning_executor([response["tokens"]], response["prompt_table"], '0', None)[0]

request = tensorrt_llm.bindings.executor.Request(
                input_token_ids=response["tokens"], # not none
                max_tokens=32,
                pad_id=tokenizer.pad_token_id,
                end_id=tokenizer.eos_token_id,
                output_config=output_config, # should not be none
                prompt_tuning_config=ptuning_object,
                mrope_config=mrope_config,
            )

request_ids = trt_model.session.enqueue_requests([request])
multi_responses = trt_model.session.await_responses(request_ids)
responses = [
    r for responses in multi_responses for r in responses
]

for r in responses:
    result = r.result
    output_token_ids = result.output_token_ids
    output_text = tokenizer.decode(output_token_ids[0])
    print(output_text)