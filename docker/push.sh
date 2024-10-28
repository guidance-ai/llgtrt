#!/bin/sh

set -x
set -e
cd $(dirname $0)/..

VER="$(sh ./llguidance/scripts/git-version.sh)"
IMAGE=llgtrt/llgtrt

docker tag llgtrt/llgtrt:latest $IMAGE:prod-$VER
docker tag llgtrt/llgtrt:dev $IMAGE:dev-$VER
docker push $IMAGE:dev-$VER
docker push $IMAGE:prod-$VER
docker push $IMAGE:dev
docker push $IMAGE:latest
