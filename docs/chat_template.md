# Chat templates

The chat templates in llgtrt use Jinja2 syntax, and aim to mimic the
semantics of the HF Transformers `chat_template` field in `tokenizer_config.json` file.

The chat template is used to format an array of messages into a single string.
It is also used to inject JSON schemas into the prompt based on `response_format`
and `tools` fields.

## REST access

From the user side, you can disable injection of the JSON schema by setting
`"include_json_schema_in_prompt": false` at the top-level of object POSTed
to the `/v1/chat/completions` endpoint.
You can additionally set `"return_expanded_prompt": true` to see the expanded
prompt with the JSON schema injected.

Here's an example JSON for `/v1/chat/completions`:

```json
{
  "model": "model",
  "messages": [
    { "role": "system",
      "content": "You are a helpful assistant." },
    { "role": "user",
      "content": "Please tell me a one line joke." }
  ],
  "return_expanded_prompt": true,
  "include_json_schema_in_prompt": true,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "joke": {
            "type": "string"
          },
          "rating": {
            "type": "number"
          }
        },
        "additionalProperties": false,
        "required": ["joke", "rating"]
      }
    }
  },
  "max_tokens": 100,
  "temperature": 1.2
}
```

And here's an example response (the quality of the joke may vary):

```json
{
  "id": "cmpl-3548a3c2-845f-4ce7-ac86-855c3661f0f9",
  "object": "text_completion",
  "created": 1730250367,
  "model": "model",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"joke\": \"What do you call fake spaghetti? An impasta.\"  , \"rating\": 0.8}"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 49,
    "completion_tokens": 26,
    "total_tokens": 75
  },
  "expanded_prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 30 October 2024\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nAnswer the user's question following this exact JSON schema:\n{\n    \"type\": \"object\",\n    \"properties\": {\n        \"joke\": {\n            \"type\": \"string\"\n        },\n        \"rating\": {\n            \"type\": \"number\"\n        }\n    },\n    \"additionalProperties\": false,\n    \"required\": [\n        \"joke\",\n        \"rating\"\n    ]\n}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease tell me a one line joke.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}
```

## Configuring

