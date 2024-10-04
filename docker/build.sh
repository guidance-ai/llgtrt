#!/bin/sh

cd $(dirname $0)/..

if [ ! -f TensorRT-LLM/README.md ] ; then git submodule update --init ; fi

if [ -z "$1" ]; then
    IMAGE=llgtrt_prod
else
    IMAGE="$1"
    shift
fi

docker build --progress=plain -t $IMAGE --target $IMAGE . -f docker/Dockerfile
