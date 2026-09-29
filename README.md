# CUDA Parallel Operators

一个用于练习 CUDA C++、资源管理和性能分析的入门项目。项目实现了向量加法、二维矩阵加法和共享内存分块矩阵乘法，并使用 CUDA Event 测量内核执行时间。

## 项目亮点

- 使用一维 grid-stride loop 实现模板化向量加法。
- 使用二维 Grid/Block 实现矩阵加法和边界检查。
- 使用共享内存分块实现矩阵乘法，减少全局显存重复访问。
- 使用 `CudaBuffer<T>` 和 `CudaStream` 通过 RAII 自动管理 CUDA 资源。
- 禁止资源对象复制并支持移动语义，避免重复释放。
- 使用 CUDA Event 完成 GPU 内核预热与平均耗时测量，并比较 128、256、512 三种线程块大小。
- 对所有 GPU 结果执行 CPU 参考结果验证。
- 输出向量加法显存带宽和矩阵乘法 GFLOP/s。

## 目录结构

```text
cuda-parallel-operators/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── cuda_buffer.cuh
│   ├── cuda_check.cuh
│   └── cuda_stream.cuh
└── src/
    └── main.cu
```

## 环境要求

- NVIDIA GPU
- CUDA Toolkit 12.x 或更高版本
- 支持 C++17 的主机编译器
- 可选：CMake 3.20+
- 可选：Nsight Systems、Nsight Compute

## 编译

### 直接使用 NVCC

```bash
nvcc -std=c++17 -O3 -lineinfo src/main.cu -Iinclude -o cuda_operators
```

运行：

```bash
./cuda_operators
```

### 使用 CMake

```bash
cmake -S . -B build
cmake --build build -j
./build/cuda_operators
```

如果 CMake 无法自动确定 GPU 架构，可以显式指定。例如 RTX 4090 对应 `89`：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

## 示例输出

不同 GPU 的性能数据会不同：

```text
CUDA Parallel Operators
GPU: NVIDIA GeForce RTX xxxx

[Vector Add]
  Elements: 1048576
  Block 128: ... ms, ... GB/s
  Block 256: ... ms, ... GB/s
  Block 512: ... ms, ... GB/s
  Correct:  yes
  Selected 256-thread time: ... ms

[Matrix Add]
  Shape:   1024 x 1024
  Correct: yes
  Kernel:  ... ms

[Tiled Matrix Multiply]
  Shape:       256 x 256 x 256
  Tile size:   16 x 16
  Correct:     yes
  CPU time:    ... ms
  GPU kernel:  ... ms
  Kernel speedup: ...x
  Throughput:  ... GFLOP/s
```

`Kernel speedup` 只比较 CPU 计算时间与 GPU 内核时间，不包含主机和设备之间的数据传输，因此不能代表完整应用的端到端加速比。

## Nsight Systems

分析数据传输、内核执行和同步关系：

```bash
mkdir -p reports
nsys profile -t cuda,nvtx -o reports/operators ./cuda_operators
```

生成报告后可以执行：

```bash
nsys stats reports/operators.nsys-rep
```

建议观察：

- H2D/D2H 数据传输耗时；
- 三个内核的执行顺序和耗时；
- `cudaStreamSynchronize` 带来的等待；
- `cudaMalloc` 和 `cudaFree` 的调用位置。

## Nsight Compute

分析每个内核的访存和硬件利用率：

```bash
mkdir -p reports
ncu --set full -o reports/operators_compute ./cuda_operators
```

建议重点观察：

- `Occupancy`
- `DRAM Throughput`
- `Memory Throughput`
- `Warp Execution Efficiency`
- `Shared Memory Throughput`

部分服务器会禁止普通用户访问 GPU Performance Counter。如果出现 `ERR_NVGPUCTRPERM`，需要管理员在宿主机驱动层开放性能计数器权限。

## 实现说明

### 向量加法

每个线程使用 grid-stride loop 处理一个或多个元素。该算子通常受显存带宽限制，因此程序根据两次读取和一次写入估算有效带宽。

### 矩阵加法

使用二维线程块映射矩阵的行和列，并在写入前检查边界。

### 分块矩阵乘法

每个线程块将 A、B 的局部 Tile 加载到共享内存，同一数据可被块内多个线程复用。每轮计算前后使用 `__syncthreads()` 保证数据加载和使用完成。

当前实现用于展示 CUDA 优化思想，不用于替代 cuBLAS。在实际工程中，通用矩阵乘法应优先使用 cuBLAS 或 CUTLASS。

## 后续改进

- 增加朴素矩阵乘法，展示共享内存优化前后的对比。
- 增加不同 Block/Tile 大小的自动测试。
- 使用页锁定内存和多 Stream 实现数据传输与计算重叠。
- 与 `cublasSgemm` 进行性能和正确性对比。
- 添加 GitHub Actions 编译检查。
