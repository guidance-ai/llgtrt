#!/bin/sh

cd $(dirname $0)/..
./docker/drun.sh "$@" llgtrt/llgtrt:latest /bin/bash -l
