import pytest
import json
import sys

sys.path.append(".")
from scripts.req import send_one, send_one_stream, req_data, messages

json_tag = "<|python_tag|>"
n = 10


def joke_data():
    d = req_data(temperature=1.2, prompt_size=0, llg=False, max_tokens=50)
    d["n"] = n
    return d


def send_and_check(d):
    resp = send_one(d)
    assert resp is not None
    assert len(resp["choices"]) == n
    return resp


def test_simple_chat():
    resp = send_and_check(joke_data())
    jokes = [c["message"]["content"] for c in resp["choices"]]
    assert len(set(jokes)) > 1


def test_stream_chat():
    resp = send_one_stream(joke_data())
    assert len(resp) == n
    jokes = [r.text for r in resp]
    assert len(set(jokes)) > 1


def test_resp_format_json_object():
    d = {
        "model": "model",
        "messages": messages("Please tell me a one line joke."),
        "max_tokens": 50,
        "temperature": 0.8,
        "n": n,
        "response_format": {"type": "json_object"},
    }
    resp = send_and_check(d)

    num_ok = 0
    for c in resp["choices"]:
        content: str = c["message"]["content"]
        if c["finish_reason"] == "stop":
            d = json.loads(content)
            num_ok += 1

    assert num_ok > 0


def schema_data(msg: str, schema: dict):
    return {
        "model": "model",
        "messages": messages(msg),
        "max_tokens": 50,
        "temperature": 0.8,
        "n": n,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": schema,
            },
        },
    }


def send_schema(msg: str, schema: dict):
    d = schema_data(msg, schema)
    return send_and_check(d)


def test_resp_format_json_schema():
    resp = send_schema(
        "Please tell me a one line joke.",
        {
            "type": "object",
            "properties": {
                "joke": {"type": "string"},
                "rating": {"type": "number"},
            },
            "additionalProperties": False,
            "required": ["joke", "rating"],
        },
    )

    num_ok = 0
    for c in resp["choices"]:
        content: str = c["message"]["content"]
        if c["finish_reason"] == "stop":
            j = json.loads(content)
            assert "joke" in j
            assert "rating" in j
            num_ok += 1

    assert num_ok > 0


def test_resp_string_schema():
    resp = send_schema(
        "How much is 1+1?",
        {
            "$schema": "http://json-schema.org/draft-06/schema#",
            "type": "string",
            "enum": ["1", "2-two", "3 three", None],
        },
    )
    for c in resp["choices"]:
        content: str = c["message"]["content"]
        assert content ==  '"2-two"'


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("weather", [True, False])
def test_tools(strict: bool, weather: bool):
    d = {
        "model": "model",
        "messages": messages(
            "What is the weather in London?" if weather else "How much is 2 + 2?"
        ),
        "max_tokens": 50,
        "temperature": 0.8,
        "n": n,
        "tool_choice": "required" if weather else "auto",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get the weather for a <location>",
                    "strict": strict,
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "additionalProperties": False,
                        "required": ["location"],
                    },
                },
            }
        ],
    }
    resp = send_and_check(d)
    for c in resp["choices"]:
        content: str = c["message"]["content"]
        if weather:
            assert '"function"' in content
            assert content.startswith(json_tag)
            j = json.loads(content[len(json_tag) :])
            assert j["name"] == "weather"
            assert "London" in j["parameters"]["location"]
        else:
            assert '"function"' not in content
