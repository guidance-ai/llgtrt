# This argument allows you to override the base image with your own custom build of TensorRT or TensorRT-LLM
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.03-py3

# Set this to false if your image already includes TensorRT-LLM
ARG INSTALL_TRTLLM=true

# Set this to 1 to enable the CXX11_ABI during native builds
ARG USE_CXX11_ABI=0

FROM ${BASE_IMAGE} AS llgtrt_dev

ARG INSTALL_TRTLLM
ARG USE_CXX11_ABI

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache \
    cmake curl libjpeg-dev libpng-dev strace \
    llvm-dev libclang-dev clang ccache apache2-utils git-lfs \
    screen bsdmainutils pip python3-dev python3-venv python-is-python3 \
    pkg-config software-properties-common linux-tools-common

#    openmpi-bin libopenmpi-dev

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.83.0

RUN curl https://sh.rustup.rs -sSf | sh -s -- \
     -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION
RUN rustup component add rustfmt
RUN cargo install rustfilt

# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
# ENV VIRTUAL_ENV="/opt/venv"

# install tensorrt_llm packages
RUN mkdir -p /tmp/reqs

# if this fails, run 'git submodule update --init' first

COPY TensorRT-LLM/examples/constraints.txt /tmp/reqs/requirements.txt
# COPY TensorRT-LLM/requirements.txt /tmp/reqs/
# COPY TensorRT-LLM/docker/common/install_mpi4py.sh /tmp/reqs/
# COPY TensorRT-LLM/docker/common/install_tensorrt.sh /tmp/reqs/

#RUN bash /tmp/reqs/install_mpi4py.sh
#RUN bash /tmp/reqs/install_tensorrt.sh

# Install tensorrt_llm and its backing c++ libraries from pip unless disabled
RUN echo "INSTALL_TRTLLM is set to: ${INSTALL_TRTLLM}" && \
    if [ "${INSTALL_TRTLLM}" = "true" ]; then \
        cd /tmp/reqs && \
        pip install \
            --extra-index-url https://pypi.nvidia.com/ \
            --upgrade -r requirements.txt && \
        rm -rf /root/.cache/pip; \
    fi

# more packages for this image
RUN pip install \
 torch transformers bitsandbytes \
 pytest ujson numpy requests mypy maturin[zig] \
 guidance huggingface_hub tokenizers jsonschema \
 pandas matplotlib plotly wheel && \
 pip uninstall -y guidance && \
 rm -rf /root/.cache/pip

# TODO test this
# RUN pip install flash-attn --no-build-isolation
# RUN pip install qwen_vl_utils

RUN cd /usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/ && \
    ln -s libnvinfer_plugin_tensorrt_llm.so libnvinfer_plugin_tensorrt_llm.so.10

# dial down start banner a bit
RUN rm -f /opt/nvidia/entrypoint.d/40-tensorrt-samples-info.txt
# RUN rm -f /opt/nvidia/entrypoint.d/10-banner.sh
RUN rm -f /opt/nvidia/entrypoint.d/29-tensorrt-url.txt

USER root

FROM llgtrt_dev AS llgtrt_builder

RUN mkdir -p /workspaces/llgtrt
WORKDIR /workspaces/llgtrt
COPY . .

# link stub, so that it builds without nvidia-runtime
RUN cd /usr/local/cuda/lib64 && ln -s stubs/libnvidia-ml.so libnvidia-ml.so.1
RUN bash -c 'env'
RUN ./scripts/build.sh --clean --cxx11abi $USE_CXX11_ABI
# remove stub just in case
RUN rm /usr/local/cuda/lib64/libnvidia-ml.so.1

FROM llgtrt_dev AS llgtrt_prod

COPY --from=llgtrt_builder /workspaces/llgtrt/target/release/llgtrt /usr/local/bin/llgtrt
COPY --from=llgtrt_builder /workspaces/llgtrt/TensorRT-LLM/examples /opt/TensorRT-LLM-examples
COPY --from=llgtrt_builder /workspaces/llgtrt/scripts/launch-llgtrt.sh /usr/local/bin/launch-llgtrt.sh
