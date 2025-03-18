extern "C" __global__ void matmul_transb(const float* A, const float* B, float* C, float beta, float alpha, int m, int n, int k) {
    // 共享内存大小
    const int BLOCK_SIZE = 16;
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE + 1]; // 添加 padding 避免 bank conflict
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE + 1]; // 添加 padding 避免 bank conflict

    // 线程索引
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    // 遍历 K 维度
    for (int i = 0; i < k; i += BLOCK_SIZE) {
        // 计算当前块的偏移
        int load_a_row = row;
        int load_a_col = i + threadIdx.y;

        // 加载 A 的数据到共享内存（非向量化）
        if (load_a_row < m && load_a_col < k) {
            shared_A[threadIdx.x][threadIdx.y] = A[load_a_row * k + load_a_col];
        } else {
            shared_A[threadIdx.x][threadIdx.y] = 0.0f;
        }

        // 加载 B^T 的数据到共享内存（非向量化）
        int load_b_row = i + threadIdx.y; // B^T 的行索引（即 B 的列索引）
        int load_b_col = col;            // B^T 的列索引（即 B 的行索引）
        if (load_b_row < k && load_b_col < n) {
            shared_B[threadIdx.x][threadIdx.y] = B[load_b_row * n + load_b_col];
        } else {
            shared_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // 计算部分和
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += shared_A[threadIdx.x][j] * shared_B[threadIdx.y][j]; // 利用 padding 减少 bank conflict
        }

        __syncthreads();
    }

    // 写入结果
    if (row < m && col < n) {
        C[row * n + col] = beta * C[row * n + col] + alpha * sum;
    }
}
