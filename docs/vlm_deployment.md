# Tutorial to deploy llama-3.2-vision-instruct

### Download the model
```
cd llgtrt
mkdir models && cd models
git clone https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
```

### Build TRT Engine
Check this for instruction to build trt engine for mllama: https://github.com/NVIDIA/TensorRT-LLM/tree/v0.18.0/examples/multimodal#mllama

Launch llgtrt container to build the engine
```
docker run -it --gpus=all -v `pwd`:/models llgtrt/llgtrt bash
```

Pick the right max_num_tokens, max_seq_len and max_batch_size for your needs
```
python /opt/TensorRT-LLM-examples/mllama/convert_checkpoint.py --model_dir /models/Llama-3.2-11B-Vision-Instruct --output_dir /tmp/mllama/trt_ckpts --dtype bfloat16

python3 -m tensorrt_llm.commands.build \
            --checkpoint_dir /tmp/mllama/trt_ckpts \
            --output_dir /models/trt_engines/decoder/ \
            --max_num_tokens 4096 \
            --max_seq_len 2048 \
            --workers 1 \
            --gemm_plugin bfloat16 \
            --max_batch_size 4 \
            --max_encoder_input_len 6404 \
            --input_timing_cache model.cache

cp /models/Llama-3.2-11B-Vision-Instruct/tokenizer* /models/trt_engines/decoder/
cp /models/Llama-3.2-11B-Vision-Instruct/preprocessor*.json /models/trt_engines/decoder/
```

Create llgtrt.json5 file
```
llgtrt --engine /models/trt_engines/decoder --print-config > /models/trt_engines/decoder/llgtrt.json5
```
If above command fails, just run "llgtrt --engine /models/trt_engines/decoder --print-config" and copy the json output to /models/trt_engines/decoder/llgtrt.json5

Move HF model into engine directory
```
mv /models/Llama-3.2-11B-Vision-Instruct /models/trt_engines/decoder/hf_model
```

Exit the container

### Prepare preprocessing script for images
```
cd llgtrt
cp llgtrt/py/llama_3.2_vision/input_processor.py models/trt_engines/decoder/
```

Edit llgtrt.json5 file to modify Python API session
```
//"input_processor": null, -> "input_processor": "/engine/input_processor.py",
//"hf_model_dir": null, -> "hf_model_dir": "/engine/hf_model",
```
By default, when you run the llgtrt container, the engine directory will be mapped to /engine directory

Edit llgtrt.json5 file to modify runtime session
```
//"max_batch_size": 128, -> "max_batch_size": 4,
//"cross_kv_cache_fraction": null -> "cross_kv_cache_fraction": 0.4,
```
Pick the right values for your needs, 
we used batch_size 4 in above example when we built the engine

If you are using bfloat16 engine as above, edit input_processor.py file in models/trt_engines/decoder, go to line 88 and change
```
r.encoder_input_features = cross_attention_states.cuda().half() 
to 
r.encoder_input_features = cross_attention_states.cuda().bfloat16()
```

### Launch the container
```
sudo PORT=3000 ./docker/run.sh `pwd`/models/trt_engines/decoder
```

Run a test
```
cd llgtrt/scripts
bash test-infer.sh vlm
```
