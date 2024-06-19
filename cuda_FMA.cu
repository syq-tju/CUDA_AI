#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 16; // 线程块大小

// CUDA Kernel函数：矩阵乘法（使用FMA指令）
__global__ void matrixMultiplyFMA(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            float a = A[row * N + k];
            float b = B[k * N + col];
            sum = fma(a, b, sum); // 使用FMA指令进行乘法融合加法
        }
        C[row * N + col] = sum;
    }
}

// 初始化矩阵数据（随机初始化）
void initializeMatrix(float *matrix, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f; // 随机数范围 [0, 10)
    }
}

// 输出部分矩阵元素用于验证结果
void printMatrix(float *matrix, int size, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int N = 1024; // 矩阵大小 NxN
    size_t size = N * N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrixMultiplyFMA<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Matrix C (partial):" << std::endl;
    printMatrix(h_C, N, 4, 4); // 输出前4行4列的矩阵 C 的元素

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
