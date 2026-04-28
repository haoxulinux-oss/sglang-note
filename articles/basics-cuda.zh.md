# CUDA 是什么

---

## 一 全称与定位

**CUDA** = **Compute Unified Device Architecture**(统一计算设备架构)

NVIDIA 在 2007 年推出。它是 NVIDIA GPU 的**通用计算平台 + 编程模型 + 工具链**——让你能用 GPU 跑「不只是图形」的并行计算任务,比如深度学习、科学计算、密码学、视频编解码。

CUDA 不是单一产品,而是一整套东西:

| 层 | 内容 |
|---|---|
| 硬件 | NVIDIA GPU 的 SM(Streaming Multiprocessor)、CUDA core、Tensor core |
| 驱动 | NVIDIA Driver(`nvidia.ko` 内核模块,libcuda.so 用户态) |
| 运行时 | CUDA Runtime(libcudart.so),内存管理、stream、event、graph |
| 编程语言 | **CUDA C/C++**(`.cu` 文件,扩展了 C++ 的 GPU 关键字)、CUDA Fortran、CUDA Python |
| 编译器 | **NVCC**(把 CUDA C++ 编译成 PTX/SASS) |
| 库 | cuBLAS、cuDNN、cuFFT、NCCL、cuSPARSE、Thrust...(NVIDIA 写的高性能 GPU 库) |
| 工具 | nsight-systems、nsight-compute、cuda-gdb、nvprof |

---

## 二 GPU 通用计算的核心思想

CPU 和 GPU 设计目标相反:

| | CPU | GPU |
|---|---|---|
| 核心数 | 几个到几十个 | 几千个(简单核) |
| 单核性能 | 强、有大 cache、复杂分支预测 | 弱、共享 cache、简单调度 |
| 适合 | 顺序逻辑、复杂控制流 | 大量同质数据并行 |
| 比喻 | 4 个博士生轮流做事 | 5000 个小学生同时做加法题 |

CUDA 就是「**让程序员把任务拆成几千个并行小活,扔给 GPU 这几千个小学生同时做**」的接口。

---

## 三 一段最简单的 CUDA 代码

把两个数组相加(对 1024 个元素同时跑):

```cpp
// CUDA C++ 文件 vec_add.cu

// __global__ 标识这是个 GPU kernel(在 GPU 上跑,被 CPU 调用)
__global__ void vec_add(float* a, float* b, float* c, int n) {
    // threadIdx 和 blockIdx 是每个并行线程自带的「我是谁」编号
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *c;
    cudaMalloc(&a, 1024 * sizeof(float));   // 在 GPU 显存上分配
    cudaMalloc(&b, 1024 * sizeof(float));
    cudaMalloc(&c, 1024 * sizeof(float));

    // 把 CPU 内存的数据拷到 GPU
    cudaMemcpy(a, host_a, 1024 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, 1024 * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel:开 8 个 block,每个 block 128 个 thread = 共 1024 个线程
    vec_add<<<8, 128>>>(a, b, c, 1024);

    // 把结果拷回 CPU
    cudaMemcpy(host_c, c, 1024 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(a); cudaFree(b); cudaFree(c);
}
```

要点:

- `__global__` 函数 = **kernel**,1024 个线程会同时跑同一段代码,但每个 thread 的 `i` 不同,处理数组里不同的元素。
- `<<<grid, block>>>` 是 CUDA 独有的语法,告诉 GPU 启动多少个并行线程。
- 数据要在 CPU 和 GPU 之间显式拷贝(各有自己的内存)。
- 这个 kernel 一次启动就调度上千个并行线程——这就是 GPU 加速的来源。

---

## 四 CUDA 在深度学习中的位置

PyTorch / TensorFlow / JAX 这些深度学习框架,**底层全都是 CUDA**:

```
Python:  torch.matmul(A, B)
            │
            ▼
PyTorch C++:  调 cuBLAS::cublasSgemm
            │
            ▼
cuBLAS:   预编译的高度优化 CUDA kernel
            │
            ▼
CUDA Driver:  把 kernel 提交给 GPU
            │
            ▼
GPU:      几千个 CUDA core 并行做矩阵乘
```

你写 `output = model(x)` 看起来是一行 Python,实际背后是几十/几百次 CUDA kernel launch + cuBLAS / cuDNN 调用。

---

## 五 CUDA 关联的几个名词区分

| 名词 | 是什么 |
|---|---|
| **CUDA** | NVIDIA 的 GPU 通用计算平台总称(本文主题) |
| **CUDA Core** | NVIDIA GPU 里的标量计算单元(整数 + 浮点) |
| **Tensor Core** | NVIDIA GPU 里的矩阵乘加速单元(专为深度学习,Volta 之后引入) |
| **CUDA Toolkit** | 开发包,含 nvcc、库、调试工具 |
| **CUDA Runtime** | 进程里链接的运行时库,提供 `cudaMalloc / cudaMemcpy / cudaLaunchKernel` 等 API |
| **CUDA Driver** | 内核态驱动,管 GPU 资源调度 |
| **PTX** | CUDA 编译的中间字节码(类似 LLVM IR),不同 GPU 架构 JIT 成 SASS |
| **SASS** | 真正在 GPU 上跑的机器码 |
| **CUDA Graph** | CUDA 提供的「录制 kernel 序列后整段重放」机制 |
| **NVCC** | CUDA C/C++ 编译器 |
| **cuDNN** | NVIDIA 写的深度学习专用 CUDA 库(卷积、attention 等高性能实现) |
| **cuBLAS** | NVIDIA 写的线性代数 CUDA 库 |
| **NCCL** | 多 GPU 集合通信库(all-reduce / broadcast) |
| **Thrust** | CUDA 上的 STL-like 高层并行算法库 |

---

## 六 与对手的对比

| 方案 | 厂商 | 关系 |
|---|---|---|
| **CUDA** | NVIDIA | 闭源、只支持 NVIDIA GPU,生态最成熟 |
| **ROCm / HIP** | AMD | AMD GPU 上的 CUDA-like 平台,HIP API 几乎和 CUDA 一一对应,有 hipify 工具自动转换源码 |
| **OneAPI / SYCL** | Intel | 跨厂商开源标准,也能跑在 NVIDIA GPU 上 |
| **OpenCL** | Khronos | 跨平台开源标准,通用但 NVIDIA 不太上心,性能落后 CUDA |
| **Metal** | Apple | 苹果 GPU(M 系列芯片)的同类东西 |
| **CANN** | 华为 | 昇腾 NPU 的同类东西 |

CUDA 在深度学习领域占绝对主导,主要因为:cuDNN/cuBLAS/NCCL 性能优势 + 工具链成熟 + PyTorch 默认对接 + NVIDIA GPU 性能领先。这也是为什么「**深度学习 = NVIDIA GPU + CUDA**」长期成立。

---

## 七 与 SGLang 的关系

SGLang 是 LLM 推理引擎,自然全栈基于 CUDA:

- model.forward 跑的是一堆 CUDA kernel(matmul 走 cuBLAS / FlashInfer / FA3,attention 走 CUDA 自定义 kernel)。
- 进程里通过 PyTorch 调 CUDA Runtime API(`cudaMalloc / cudaStream / cudaEvent`)。
- 多 GPU 通信走 NCCL(也基于 CUDA)。
- decode 阶段用 **CUDA Graph** 减少 launch overhead。
- `sgl-kernel` 子目录就是 SGLang 自己写的 CUDA C++ kernel(`.cu` 文件),用 NVCC 编译成 PyTorch 扩展。
- `--device cuda`(默认) vs `--device rocm / cpu / hpu / npu`——CUDA 路径是主路径。

---

## 八 一句话总结

> **CUDA** 是 NVIDIA 的 GPU 通用计算平台:让你能用 C++ 风格的语言写出几千个线程并行运行的程序,把矩阵乘、卷积、attention 这种大量同质并行的活扔给 GPU 几千个 core 同时做。它包括硬件、驱动、运行时、编译器、库、工具链一整套。深度学习框架(PyTorch/TF) 和 LLM 推理引擎(SGLang/vLLM/TRT-LLM) 全都是建在 CUDA 之上的——所以「跑 LLM 要 NVIDIA GPU」本质上是「整个生态绑定在 CUDA 上」。
