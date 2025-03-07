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

def torch_dtype(tp_name: str) -> int: ...
