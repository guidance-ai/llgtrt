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

# Set the default base image if --trtllm was not provided
if [[ -z "$BASE_IMAGE" ]]; then
    BASE_IMAGE="nvcr.io/nvidia/tensorrt:24.12-py3"
fi

if $DEV_MODE; then
    TARGET="--target llgtrt_dev"
else
    TARGET="--target llgtrt_prod"
fi

# Build the Docker image with the appropriate arguments
DOCKER_BUILD_ARGS="--progress=plain --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg INSTALL_TRTLLM=$INSTALL_TRTLLM --build-arg USE_CXX11_ABI=$USE_CXX11_ABI $TARGET"

echo "Building Docker image $TARGET with arguments: $DOCKER_BUILD_ARGS"
docker build $DOCKER_BUILD_ARGS . -f docker/Dockerfile
