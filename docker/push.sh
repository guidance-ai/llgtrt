#!/bin/sh

set -x
set -e
cd $(dirname $0)/..

VER="$(sh ./llguidance/scripts/git-version.sh)"
# check if $1 is in user/repo format
case "$1" in
  */*)
    IMAGE="$1"
    shift
    ;;
  *)
    echo "Usage: $0 user/repo"
    exit 1
    ;;
esac

docker tag llgtrt_prod $IMAGE:prod-$VER
docker tag llgtrt_dev $IMAGE:dev-$VER
docker push $IMAGE:dev-$VER
docker push $IMAGE:prod-$VER
