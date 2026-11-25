cmake_minimum_required(VERSION 3.22)

project(flow CXX CUDA)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

FetchContent_Declare(cuda-api-wrappers_library
        GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
        GIT_TAG v0.8.1)
FetchContent_MakeAvailable(cuda-api-wrappers_library)

FetchContent_Declare(raylib_library
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 5.5)
FetchContent_MakeAvailable(raylib_library)

set(CUDA_FLAGS
        --expt-extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        --ptxas-options -v
        --generate-line-info
)

add_library(flow STATIC src/gl_interop.h src/gl_interop.cpp)

target_compile_options(flow PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
)

target_link_libraries(flow cuda-api-wrappers::runtime-and-driver raylib)

target_include_directories(flow PUBLIC src)

add_subdirectory(examples)

