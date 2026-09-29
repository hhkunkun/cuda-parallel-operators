#pragma once

#include "cuda_check.cuh"

#include <cstddef>
#include <utility>

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&data_, bytes()));
        }
    }

    ~CudaBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                cudaFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void upload(const T* host_data, cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(
            data_, host_data, bytes(), cudaMemcpyHostToDevice, stream));
    }

    void download(T* host_data, cudaStream_t stream = nullptr) const {
        CUDA_CHECK(cudaMemcpyAsync(
            host_data, data_, bytes(), cudaMemcpyDeviceToHost, stream));
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

