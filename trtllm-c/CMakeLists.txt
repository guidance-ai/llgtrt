cmake_minimum_required(VERSION 3.1)

set(CMAKE_CUDA_ARCHITECTURES 70-real 80-real 86-real 89-real 90-real)
enable_language(CUDA)

set(TRTLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../TensorRT-LLM")
set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/cpp/include")
set(TRTLLM_INCLUDE_COMMON_DIR "${TRTLLM_DIR}/cpp")

set(USE_CXX11_ABI 0)
message(STATUS "Use CXX11 ABI: ${USE_CXX11_ABI}")
add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_VERBOSE_MAKEFILE 1)

# Define project name
project(trtllm-c)

# Compile options
set(CMAKE_CXX_FLAGS "-Wall -pthread -lstdc++ -DENABLE_MULTI_DEVICE=1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_BUILD_TYPE release)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDAToolkit_VERSION}")
message(STATUS "    libraries: ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")

add_definitions("-DENABLE_BF16")
add_definitions("-DENABLE_FP8")

include_directories(${TRTLLM_INCLUDE_DIR} ${TRTLLM_INCLUDE_COMMON_DIR} ${CUDAToolkit_INCLUDE_DIRS})

add_library(trtllm_c STATIC main.cpp logits.cpp mask_logits.cu)
