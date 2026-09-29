cmake_minimum_required(VERSION 3.20)

project(cuda_parallel_operators LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_operators src/main.cu)
target_include_directories(cuda_operators PRIVATE include)

target_compile_options(cuda_operators PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(cuda_operators PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
