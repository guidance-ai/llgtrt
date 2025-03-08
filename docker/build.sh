#!/bin/bash

# Default values
DEV_MODE=false
BASE_IMAGE=""
INSTALL_TRTLLM=true
USE_CXX11_ABI=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --trtllm)
            if [[ -n "$2" && "$2" != --* ]]; then
                BASE_IMAGE="$2"
                INSTALL_TRTLLM=false
                USE_CXX11_ABI=1
                shift 2
            else
                echo "Error: --trtllm requires an argument (image name)."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

DOCKER_BUILD_ARGS="--progress=plain --build-arg INSTALL_TRTLLM=$INSTALL_TRTLLM --build-arg USE_CXX11_ABI=$USE_CXX11_ABI"

# Only override BASE_IMAGE if --trtllm is specified
if [[ -z "$BASE_IMAGE" ]]; then
    :
else
    DOCKER_BUILD_ARGS="$DOCKER_BUILD_ARGS --build-arg BASE_IMAGE=$BASE_IMAGE"
fi

function run_docker_build {
    echo "Building Docker image with arguments: $DOCKER_BUILD_ARGS $@"
    docker build $DOCKER_BUILD_ARGS "$@" -f docker/Dockerfile .
}

run_docker_build --target llgtrt_dev -t llgtrt/llgtrt:dev
if $DEV_MODE; then
    echo "Skip production build in dev mode"
else
    run_docker_build --target llgtrt_prod -t llgtrt/llgtrt:latest
fi
