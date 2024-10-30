#!/bin/sh

if [ "X$TRT_API_BASE" = "X" ] ; then
  export TRT_API_BASE="http://127.0.0.1:3000/v1/"
fi

case "$1" in
  compl)
curl -X POST "${TRT_API_BASE}completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "stream": false,
  "prompt": "Here is a one line joke: ",
  "max_tokens": 100,
  "temperature": 0.7,
  "n": 5
}' | jq
;;

  stream)
curl -X POST "${TRT_API_BASE}completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "stream": true,
  "prompt": "Here is a one line joke: ",
  "max_tokens": 100,
  "temperature": 0.7,
  "n": 5
}'
;;

  chat)
curl -X POST "${TRT_API_BASE}chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Please tell me a one line joke."
    }
  ],
  "max_tokens": 100,
  "temperature": 1.2
}' | jq
;;

  chat_min_tokens)
curl -X POST "${TRT_API_BASE}chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Please tell me a one line joke."
    }
  ],
  "max_tokens": 100,
  "min_tokens": 100,
  "temperature": 1.2
}' | jq
;;

  chat2)
  curl -X POST "${TRT_API_BASE}chat/completions" \
  -H "Content-Type: application/json" -v \
  -d '{
     "model": "model",
     "messages": [
        {"role": "system", "content": "System Message 1"},
        {"role": "user", "content": "Help me"},
        {"role": "assistant", "content": "What do you need?"},
        {"role": "system", "content": "System Message 2"}      
      ],
     "temperature": 0.7
   }'
   ;;
  
  exn)
  curl -X POST "${TRT_API_BASE}chat/completions" \
  -H "Content-Type: application/json" -v \
  -d '{
     "model": "model",
     "messages": [
        {"role": "assistant", "content": "What do you need?", "tool_calls": [{}, {}]}
      ],
     "temperature": 0.7
   }'
   ;;

  tools)
curl -X POST "${TRT_API_BASE}chat/completions" \
-H "Content-Type: application/json" -v \
-d '{
  "model": "model",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the weather in Seattle?"
    }
  ],
  "return_expanded_prompt": true,
  "include_json_schema_in_prompt": true,
  "max_tokens": 100,
  "temperature": 0.8,
  "tools": [{
    "type": "function",
    "function": {
      "name": "weather",
      "description": "Get the weather for a <location>",
      "strict": true,
      "parameters": {
         "type": "object",
         "properties": {
             "location": { "type": "string" }
         },
         "additionalProperties": false,
         "required": ["location"]
      }
    }
  }]
}' | jq
;;

  json)
curl -v -X POST "${TRT_API_BASE}chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "seed": 42,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Please tell me a one line joke."
    }
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
}' | jq
;;

  *.json)
curl -X POST "${TRT_API_BASE}chat/completions" \
-H "Content-Type: application/json" -v -d @"$1" | jq
;;

esac

echo
