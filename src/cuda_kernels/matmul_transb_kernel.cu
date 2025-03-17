extern "C" __global__ void matmul_transb(const float* A, const float* B, float* C, float beta, float alpha, int m, int n, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[col * k + i];
        }
        C[row * n + col] = beta * C[row * n + col] + alpha * sum;
    }
}
