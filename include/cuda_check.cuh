#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t error = (call);                                    \
        if (error != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)        \
                      << " at " << __FILE__ << ':' << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

