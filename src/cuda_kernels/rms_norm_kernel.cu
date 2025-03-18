extern "C" __global__ void rms_norm(const float* x, const float* w, float* y, float epsilon, int dim) {
    __shared__ float shared_sum[256]; // 假设 blockDim.x <= 256

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;

    if (idx < dim) {
        local_sum = x[idx] * x[idx];
    }

    // 归约计算 sum
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sum = shared_sum[0];
    sum = sqrtf(sum / (dim + epsilon));

    if (idx < dim) {
        y[idx] = w[idx] * x[idx] / sum;
    }
}
