#!/bin/sh

if [ -z "$PORT" ]; then
    PORT=3000
fi

# if [ -z "$1" ]; then
#     echo "Usage: $0 /path/to/engine [/path/to/lora/weights]"
#     exit 1
# else
#     ENGINE="$1"
#     shift
# fi

# if [ ! -z "$1" ]; then
#     LORADIR="$1"
#     shift
# fi

# if test -f "$ENGINE/rank0.engine" ; then
#     :
# else
#     echo "Error: $ENGINE/rank0.engine not found - doesn't look like engine directory"
#     exit 1
# fi
MODEL_DIR="/media/nvme1n1/models"

lora_volume=''
lora_arg=''
if [ ! -z "$LORADIR" ]; then
    lora_volume="--volume $LORADIR:/lora"
    lora_arg="--lora-root /lora"
fi

set -ex
cd $(dirname $0)/..
cd $(dirname $0)/..

docker run \
    --volume "$MODEL_DIR":/models \
    --ipc=host \
    --runtime=nvidia \
    --privileged \
    --gpus=0 \
    --shm-size=8g \
    --publish $PORT:$PORT \
    -d \
    -e ENGINE=/models/8b-spec-decode-5-dt \
    -e DRAFT_ENGINE=/models/1b-spec-decode \
    -e PORT=$PORT \
    -e N_DRAFT_TOKENS=5 \
    llgtrt/llgtrt:latest \
    /usr/local/bin/launch-llgtrt-draft.sh

