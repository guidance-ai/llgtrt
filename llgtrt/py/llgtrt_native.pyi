class PluginInit:
    tokenizer_folder: str
    chat_template: str
    hf_model_dir: str

def torch_dtype(tp_name: str) -> int: ...
