#!/bin/sh

cd "$(dirname "$0")/.."
python scripts/collect-comments.py LlgTrtConfig llgtrt/src/*.rs llguidance/parser/src/*.rs
