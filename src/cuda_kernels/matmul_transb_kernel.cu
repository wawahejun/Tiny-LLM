extern "C" __global__ void matmul_transb(const float* A, const float* B, float* C, float beta, float alpha, int m, int n, int k) {
    __shared__ float shared_A[16][16];
    __shared__ float shared_B[16][16];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    for (int i = 0; i < k; i += blockDim.x) {
        // 将 A 和 B 的数据加载到共享内存
        if (row < m && (i + threadIdx.x) < k) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * k + (i + threadIdx.x)];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && (i + threadIdx.x) < k) {
            shared_B[threadIdx.y][threadIdx.x] = B[col * k + (i + threadIdx.x)];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算部分和
        for (int j = 0; j < blockDim.x; ++j) {
            sum += shared_A[threadIdx.y][j] * shared_B[threadIdx.x][j];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = beta * C[row * n + col] + alpha * sum;
    }
}
