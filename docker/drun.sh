#!/bin/sh

set -e
cd $(dirname $0)/..
set -x
docker run --ipc=host --runtime=nvidia --privileged --gpus all --shm-size=8g -it --rm "$@"
