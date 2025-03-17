extern "C" __global__ void rms_norm(const float* x, const float* w, float* y, float epsilon, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dim) {
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            sum += x[i] * x[i];
        }
        sum = sqrtf(sum / (dim + epsilon));
        y[idx] = w[idx] * x[idx] / sum;
    }
}
