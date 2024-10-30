FROM nvcr.io/nvidia/tensorrt:24.09-py3 AS llgtrt_dev

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache \
    cmake curl libjpeg-dev libpng-dev strace \
    llvm-dev libclang-dev clang ccache apache2-utils git-lfs \
    screen bsdmainutils pip python3-dev python3-venv python-is-python3 \
    pkg-config software-properties-common

#    openmpi-bin libopenmpi-dev

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.81.0

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
COPY TensorRT-LLM/examples/llama/requirements.txt /tmp/reqs/
COPY TensorRT-LLM/docker/common/install_mpi4py.sh /tmp/reqs/
COPY TensorRT-LLM/docker/common/install_tensorrt.sh /tmp/reqs/

RUN bash /tmp/reqs/install_mpi4py.sh
RUN bash /tmp/reqs/install_tensorrt.sh

RUN cd /tmp/reqs && pip install -r requirements.txt

# more packages for this image
RUN pip install \
 torch transformers bitsandbytes \
 pytest ujson numpy requests mypy maturin[zig] \
 guidance huggingface_hub tokenizers jsonschema \
 pandas matplotlib plotly wheel
RUN pip uninstall -y guidance

RUN pip install --upgrade transformers

RUN cd /usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs/ && \
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
RUN ./scripts/build.sh --clean
# remove stub just in case
RUN rm /usr/local/cuda/lib64/libnvidia-ml.so.1

FROM llgtrt_dev AS llgtrt_prod

COPY --from=llgtrt_builder /workspaces/llgtrt/target/release/llgtrt /usr/local/bin/llgtrt
COPY --from=llgtrt_builder /workspaces/llgtrt/TensorRT-LLM/examples /opt/TensorRT-LLM-examples
COPY --from=llgtrt_builder /workspaces/llgtrt/scripts/launch-llgtrt.sh /usr/local/bin/launch-llgtrt.sh
