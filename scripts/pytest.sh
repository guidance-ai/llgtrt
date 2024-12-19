#!/bin/sh

cd "$(dirname "$0")/.."
PYTHONPATH=$(pwd) pytest -v tests "$@"
