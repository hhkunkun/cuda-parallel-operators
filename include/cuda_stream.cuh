#pragma once

#include "cuda_check.cuh"

#include <utility>

class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr)) {}

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }

    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    void record(cudaStream_t stream = nullptr) const {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

inline float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
    return milliseconds;
}

