#include "cuda_buffer.cuh"
#include "cuda_check.cuh"
#include "cuda_stream.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

template <typename T>
__global__ void vector_add_kernel(const T* a, const T* b, T* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
__global__ void matrix_add_kernel(
    const T* a, const T* b, T* c, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        c[index] = a[index] + b[index];
    }
}

template <typename T, int TILE_SIZE>
__global__ void matrix_multiply_tiled_kernel(
    const T* a, const T* b, T* c, int m, int k, int n) {
    __shared__ T tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    T sum = static_cast<T>(0);

    int tile_count = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < tile_count; ++tile) {
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : static_cast<T>(0);
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : static_cast<T>(0);

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

template <typename LaunchFunction>
float benchmark_kernel(LaunchFunction launch, cudaStream_t stream, int repeats) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CudaEvent start;
    CudaEvent stop;
    start.record(stream);
    for (int i = 0; i < repeats; ++i) {
        launch();
    }
    stop.record(stream);
    stop.synchronize();
    CUDA_CHECK(cudaGetLastError());
    return elapsed_ms(start, stop) / static_cast<float>(repeats);
}

template <typename T>
bool almost_equal(const std::vector<T>& actual,
                  const std::vector<T>& expected,
                  T tolerance) {
    if (actual.size() != expected.size()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        T scale = std::max(static_cast<T>(1), std::abs(expected[i]));
        if (std::abs(actual[i] - expected[i]) > tolerance * scale) {
            std::cerr << "Mismatch at " << i << ": actual=" << actual[i]
                      << ", expected=" << expected[i] << '\n';
            return false;
        }
    }
    return true;
}

template <typename T>
std::vector<T> make_random_data(std::size_t count, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<T> distribution(
        static_cast<T>(-1), static_cast<T>(1));
    std::vector<T> values(count);
    for (T& value : values) {
        value = distribution(generator);
    }
    return values;
}

void run_vector_add(CudaStream& stream) {
    using T = float;
    constexpr int n = 1 << 20;
    constexpr int block_size = 256;
    constexpr int repeats = 100;

    auto h_a = make_random_data<T>(n, 1);
    auto h_b = make_random_data<T>(n, 2);
    std::vector<T> h_c(n);
    std::vector<T> reference(n);

    for (int i = 0; i < n; ++i) {
        reference[i] = h_a[i] + h_b[i];
    }

    CudaBuffer<T> d_a(n);
    CudaBuffer<T> d_b(n);
    CudaBuffer<T> d_c(n);
    d_a.upload(h_a.data(), stream.get());
    d_b.upload(h_b.data(), stream.get());
    stream.synchronize();

    const int block_sizes[] = {128, 256, 512};
    float milliseconds = 0.0f;

    std::cout << "[Vector Add]\n"
              << "  Elements: " << n << '\n';

    for (int current_block_size : block_sizes) {
        int grid_size = std::min(
            (n + current_block_size - 1) / current_block_size, 4096);
        auto launch = [&] {
            vector_add_kernel<T>
                <<<grid_size, current_block_size, 0, stream.get()>>>(
                    d_a.data(), d_b.data(), d_c.data(), n);
        };

        float current_ms = benchmark_kernel(launch, stream.get(), repeats);
        double moved_bytes = 3.0 * n * sizeof(T);
        double bandwidth = moved_bytes / (current_ms * 1.0e6);

        std::cout << "  Block " << std::setw(3) << current_block_size
                  << ": " << current_ms << " ms, "
                  << bandwidth << " GB/s\n";

        if (current_block_size == block_size) {
            milliseconds = current_ms;
        }
    }

    d_c.download(h_c.data(), stream.get());
    stream.synchronize();

    bool correct = almost_equal(h_c, reference, static_cast<T>(1e-6));
    std::cout << "  Correct: " << (correct ? "yes" : "no") << '\n'
              << "  Selected 256-thread time: " << milliseconds << " ms\n\n";
}

void run_matrix_add(CudaStream& stream) {
    using T = float;
    constexpr int rows = 1024;
    constexpr int cols = 1024;
    constexpr int count = rows * cols;
    constexpr int repeats = 100;

    auto h_a = make_random_data<T>(count, 3);
    auto h_b = make_random_data<T>(count, 4);
    std::vector<T> h_c(count);
    std::vector<T> reference(count);
    for (int i = 0; i < count; ++i) {
        reference[i] = h_a[i] + h_b[i];
    }

    CudaBuffer<T> d_a(count);
    CudaBuffer<T> d_b(count);
    CudaBuffer<T> d_c(count);
    d_a.upload(h_a.data(), stream.get());
    d_b.upload(h_b.data(), stream.get());
    stream.synchronize();

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    auto launch = [&] {
        matrix_add_kernel<T><<<grid, block, 0, stream.get()>>>(
            d_a.data(), d_b.data(), d_c.data(), rows, cols);
    };

    float milliseconds = benchmark_kernel(launch, stream.get(), repeats);
    d_c.download(h_c.data(), stream.get());
    stream.synchronize();

    bool correct = almost_equal(h_c, reference, static_cast<T>(1e-6));
    std::cout << "[Matrix Add]\n"
              << "  Shape:   " << rows << " x " << cols << '\n'
              << "  Correct: " << (correct ? "yes" : "no") << '\n'
              << "  Kernel:  " << milliseconds << " ms\n\n";
}

void run_matrix_multiply(CudaStream& stream) {
    using T = float;
    constexpr int m = 256;
    constexpr int k = 256;
    constexpr int n = 256;
    constexpr int tile_size = 16;
    constexpr int repeats = 20;

    auto h_a = make_random_data<T>(m * k, 5);
    auto h_b = make_random_data<T>(k * n, 6);
    std::vector<T> h_c(m * n);
    std::vector<T> reference(m * n, 0);

    auto cpu_start = std::chrono::steady_clock::now();
    for (int row = 0; row < m; ++row) {
        for (int inner = 0; inner < k; ++inner) {
            T a_value = h_a[row * k + inner];
            for (int col = 0; col < n; ++col) {
                reference[row * n + col] += a_value * h_b[inner * n + col];
            }
        }
    }
    auto cpu_stop = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(
                        cpu_stop - cpu_start)
                        .count();

    CudaBuffer<T> d_a(m * k);
    CudaBuffer<T> d_b(k * n);
    CudaBuffer<T> d_c(m * n);
    d_a.upload(h_a.data(), stream.get());
    d_b.upload(h_b.data(), stream.get());
    stream.synchronize();

    dim3 block(tile_size, tile_size);
    dim3 grid((n + tile_size - 1) / tile_size,
              (m + tile_size - 1) / tile_size);
    auto launch = [&] {
        matrix_multiply_tiled_kernel<T, tile_size>
            <<<grid, block, 0, stream.get()>>>(
                d_a.data(), d_b.data(), d_c.data(), m, k, n);
    };

    float gpu_ms = benchmark_kernel(launch, stream.get(), repeats);
    d_c.download(h_c.data(), stream.get());
    stream.synchronize();

    bool correct = almost_equal(h_c, reference, static_cast<T>(1e-3));
    double operations = 2.0 * m * n * k;
    double gflops = operations / (gpu_ms * 1.0e6);

    std::cout << "[Tiled Matrix Multiply]\n"
              << "  Shape:       " << m << " x " << k << " x " << n << '\n'
              << "  Tile size:   " << tile_size << " x " << tile_size << '\n'
              << "  Correct:     " << (correct ? "yes" : "no") << '\n'
              << "  CPU time:    " << cpu_ms << " ms\n"
              << "  GPU kernel:  " << gpu_ms << " ms\n"
              << "  Kernel speedup: " << cpu_ms / gpu_ms << "x\n"
              << "  Throughput:  " << gflops << " GFLOP/s\n\n";
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    std::cout << std::fixed << std::setprecision(3)
              << "CUDA Parallel Operators\n"
              << "GPU: " << properties.name << "\n\n";

    CudaStream stream;
    run_vector_add(stream);
    run_matrix_add(stream);
    run_matrix_multiply(stream);

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
