#!/bin/sh

if [ -z "$PORT" ]; then
    PORT=3000
fi

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/engine"
    exit 1
else
    ENGINE="$1"
    shift
fi

if test -f "$ENGINE/rank0.engine" ; then
    :
else
    echo "Error: $ENGINE/rank0.engine not found - doesn't look like engine directory"
    exit 1
fi

set -e
cd $(dirname $0)/..
./docker/drun.sh \
    --volume "$ENGINE":/engine \
    --publish $PORT:$PORT \
    llgtrt/llgtrt:latest \
    /usr/local/bin/launch-llgtrt.sh \
        /engine \
        --port $PORT \
        "$@"
