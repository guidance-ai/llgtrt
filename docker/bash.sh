#!/bin/sh

cd $(dirname $0)/..
./docker/drun.sh "$@" llgtrt_prod /bin/bash -l
