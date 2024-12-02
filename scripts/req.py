#!/usr/bin/env python3

import requests
import os
import threading
import ujson as json
import re
import random
import time
import argparse

PROMPT_SIZE = 1_00
NUM_THREADS = 10
NUM_REPS = 3
LLG = False
MAX_TOKENS = 50
NUM_JOKES = 10

TRT_API_BASE = os.getenv("TRT_API_BASE")
if TRT_API_BASE is None or TRT_API_BASE == "":
    TRT_API_BASE = "http://127.0.0.1:3000/v1/"

WORDS_STR = (
    "TheBeToOfAndAInThatHaveIItForNotOnWithHeAsYouDoAtThisButHisByFrom"
    + "TheyWeSayHerSheOrAnWillMyOneAllWouldThereTheirWhatSoUpOutIf"
    + "AboutWhoGetWhichGoMe"
)


def gen_prompt(n_tokens: int) -> str:
    words = [s.lower() for s in re.findall(r"[A-Z][a-z]*", WORDS_STR)]
    return " ".join(random.choices(words, k=n_tokens))


def messages(user_msg: str, system_msg: str = "You're a helpful assistant"):
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def send_req(path: str, payload: dict):
    url = f"{TRT_API_BASE}{path}"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def send_stream(path: str, payload: dict) -> requests.Response | None:
    url = f"{TRT_API_BASE}{path}"
    headers = {"Content-Type": "application/json"}
    payload["stream"] = True
    response = requests.post(url, headers=headers, json=payload, stream=True)
    if response.status_code == 200:
        return response
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def joke_msg():
    return messages(
        "Ignore the text below.\n"
        + gen_prompt(PROMPT_SIZE)
        + "\n\n"
        + "Now, please tell me a long joke."
    )


def llg_data():
    return {
        "controller": "llguidance",
        "controller_arg": {
            "grammar": json.loads(grammar),
        },
        "messages": joke_msg(),
        "max_tokens": MAX_TOKENS,
        "temperature": 0.8,
    }


def req_data():
    properties = {}
    required = []
    for idx in range(NUM_JOKES):
        properties[f"joke_{idx}"] = {"type": "string"}
        properties[f"rating_{idx}"] = {"type": "number"}
        required.extend([f"joke_{idx}", f"rating_{idx}"])
    return {
        "model": "model",
        "messages": joke_msg(),
        ("response_format" if LLG else "ignore_me"): {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": False,
                    "required": required,
                },
            },
        },
        # "llg_log_level": "json",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.8,
    }


class Results:
    def __init__(self) -> None:
        self.ttft = 0
        self.tbt = []
        self.usage = {}
        self.text_chunks = []
        self.logs = ""
        self.error = ""

    def finalize(self):
        self.prompt_tokens = self.usage.get("prompt_tokens", 0)
        self.completion_tokens = self.usage.get("completion_tokens", 0)
        self.completion_tokens2 = len(self.tbt)
        self.text = "".join(self.text_chunks)
        print(self.text)
        if not self.tbt:
            self.avg_tbt = 0
            self.med_tbt = 0
        else:
            self.avg_tbt = sum(self.tbt) // len(self.tbt)
            tbt_sorted = sorted(self.tbt)
            self.p50_tbt = tbt_sorted[len(self.tbt) // 2]
            self.p90_tbt = tbt_sorted[len(self.tbt) * 9 // 10]

    def __repr__(self) -> str:
        return f"ttft:{self.ttft}/{self.prompt_tokens} tbt: p50:{self.p50_tbt} p90:{self.p90_tbt} tokens:{self.completion_tokens}"


responses: list[Results] = []


def to_micros(t: float) -> int:
    return int(t * 1_000_000)


def send_one(data):
    response = send_req("chat/completions", data)
    return response


def send_one_stream(data: dict) -> list[Results]:
    t0 = time.monotonic()
    path = "chat/completions"
    is_run = False
    if data.get("controller", None) == "llguidance":
        path = "run"
        is_run = True
    response = send_stream(path, data)
    if not response:
        return []
    t_prev = None
    n = data.get("n", 1)
    results = [Results() for _ in range(n)]
    for line in response.iter_lines():
        # print type of line
        line: bytes
        if line:
            if not line.startswith(b"data: "):
                print(line)
                continue

            line = line[len(b"data: ") :]
            if line == b"[DONE]":
                break
            data = json.loads(line)
            if data.get("error", None):
                print("ERROR", data)
                results[0].error = json.dumps(data["error"])
                break

            if data["object"] == "initial-run":
                continue

            idx: int = (
                data["choices"][0]["index"] if not is_run else data["forks"][0]["index"]
            )
            res = results[idx]

            now = time.monotonic()
            if res.ttft == 0:
                res.ttft = to_micros(now - t0)
            if t_prev is not None:
                res.tbt.append(to_micros(now - t_prev))
            t_prev = now

            res.usage = data["usage"]
            if is_run:
                res.text_chunks.append(data["forks"][0]["text"])
                res.logs += data["forks"][0]["logs"]
                res.error = data["forks"][0].get("error", "")
            else:
                res.text_chunks.append(data["choices"][0]["delta"]["content"])
                res.logs += data["choices"][0].get("llg_logs", "")
    for res in results:
        res.finalize()
    responses.extend(results)
    return results


def send_a_few():
    for i in range(NUM_REPS):
        send_one_stream(req_data())


def one_round():
    global responses
    responses = []
    threads = []
    for i in range(NUM_THREADS):
        thread = threading.Thread(target=send_a_few, args=())
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    num_responses = len(responses)
    avg_p50 = sum(r.p50_tbt for r in responses) // num_responses
    avg_p90 = sum(r.p90_tbt for r in responses) // num_responses
    avg_tokens = sum(r.completion_tokens for r in responses) // num_responses
    avg_ttft = sum(r.ttft for r in responses) // num_responses

    print(
        f"# avg_p50: {avg_p50} avg_p90: {avg_p90} ({avg_tokens} tokens) avg_ttft: {avg_ttft} ({num_responses} samples, {NUM_THREADS} threads, {NUM_REPS} reps, LLG={LLG})"
    )

    return [avg_p50, avg_p90, avg_ttft, avg_tokens], [NUM_THREADS, NUM_REPS, LLG]

    # print(responses[0]["choices"][0].get("llg_logs", ""))


grammar = r"""
{"grammars":[{"nodes":[{"Join":{"sequence":[9],"max_tokens":null,"name":null,"capture_name":null}},{"String":{"literal":"{","max_tokens":null,"name":null,"capture_name":null}},{"String":{"literal":"\"joke\"","max_tokens":null,"name":null,"capture_name":null}},{"String":{"literal":":","max_tokens":null,"name":null,"capture_name":null}},{"Lexeme":{"rx":"\"(\\\\([\\\"\\\\\\/bfnrt]|u[a-fA-F0-9]{4})|[^\\\"\\\\\\x00-\\x1F\\x7F])*\"","contextual":null,"temperature":null,"json_string":false,"json_allowed_escapes":null,"json_raw":null,"max_tokens":null,"name":null,"capture_name":null}},{"String":{"literal":",","max_tokens":null,"name":null,"capture_name":null}},{"String":{"literal":"\"rating\"","max_tokens":null,"name":null,"capture_name":null}},{"Lexeme":{"rx":"-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?","contextual":null,"temperature":null,"json_string":false,"json_allowed_escapes":null,"json_raw":null,"max_tokens":null,"name":null,"capture_name":null}},{"String":{"literal":"}","max_tokens":null,"name":null,"capture_name":null}},{"Join":{"sequence":[1,2,3,4,5,6,3,7,8],"max_tokens":null,"name":null,"capture_name":null}}],"greedy_lexer":false,"greedy_skip_rx":"[\\x20\\x0A\\x0D\\x09]+","contextual":null,"rx_nodes":[],"allow_initial_skip":false,"no_forcing":false,"allow_invalid_utf8":false}],"max_tokens":null,"test_trace":false}
"""


def main():
    # random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_threads", type=int, default=0)
    parser.add_argument("--sessions", type=int, default=0)
    args = parser.parse_args()

    global LLG, NUM_REPS, NUM_THREADS, MAX_TOKENS, NUM_JOKES, PROMPT_SIZE

    if args.sessions > 0:
        LLG = True
        NUM_THREADS = args.sessions
        PROMPT_SIZE = 2600
        PROMPT_SIZE = 40_000
        NUM_REPS = 1
        NUM_JOKES = 100
        MAX_TOKENS = 4000
        one_round()
        return


    if args.max_threads > 0:
        thr = 1

        def csv_line(lst):
            print(",".join(str(x) for x in lst))

        csv_line(
            [
                "threads",
                "reps",
                "p50_tbt",
                "p90_tbt",
                "ttft",
                "tokens",
                "p50_tbt (llg)",
                "p90_tbt (llg)",
                "ttft (llg)",
                "tokens (llg)",
            ]
        )

        while thr <= args.max_threads:
            NUM_THREADS = thr
            LLG = False
            r0, k0 = one_round()
            LLG = True
            r1, k1 = one_round()
            k0.pop()
            k1.pop()
            assert k0 == k1
            csv_line(k0 + r0 + r1)
            thr *= 2

        return

    # d = llg_data()
    d = req_data()
    d["n"] = 1
    d["temperature"] = 1.0
    d["max_tokens"] = 10
    # d["logprobs"] = True
    # d["stop"] = ["Xked", "d ask"]
    if False:
        print(send_one(d))
    else:
        d["llg_log_level"] = "json"
        for r in send_one_stream(d):
            # print(r.text)
            print(repr(r))
            print(r.logs)
            print(r.error)


main()
