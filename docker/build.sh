#!/bin/sh

cd $(dirname $0)/..

if [ ! -f TensorRT-LLM/README.md ] ; then git submodule update --init ; fi

docker build --progress=plain -t llgtrt/llgtrt:dev --target llgtrt_dev . -f docker/Dockerfile

if [ "$1" != "--dev" ] ; then
    docker build --progress=plain -t llgtrt/llgtrt:latest --target llgtrt_prod . -f docker/Dockerfile
fi
