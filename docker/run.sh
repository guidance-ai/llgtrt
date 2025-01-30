#!/bin/sh

if [ -z "$PORT" ]; then
    PORT=3000
fi

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/engine [/path/to/lora/weights]"
    exit 1
else
    ENGINE="$1"
    shift
fi

if [ ! -z "$1" ]; then
    LORADIR="$1"
    shift
fi

if test -f "$ENGINE/rank0.engine" ; then
    :
else
    echo "Error: $ENGINE/rank0.engine not found - doesn't look like engine directory"
    exit 1
fi

lora_volume=''
lora_arg=''
if [ ! -z "$LORADIR" ]; then
    lora_volume="--volume $LORADIR:/lora"
    lora_arg="--lora-root /lora"
fi

set -e
cd $(dirname $0)/..
./docker/drun.sh \
    --volume "$ENGINE":/engine \
    ${lora_volume} \
    --publish $PORT:$PORT \
    llgtrt/llgtrt:latest \
    /usr/local/bin/launch-llgtrt.sh \
        /engine \
        --port $PORT \
        ${lora_arg} \
        "$@"
