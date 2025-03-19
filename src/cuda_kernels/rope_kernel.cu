extern "C" __global__ void rope(
    float* y,           // 输入输出张量
    int start_pos,      // 起始位置
    float theta,        // RoPE 参数 theta
    int seq_len,        // 序列长度
    int n_heads,        // 注意力头数
    int d               // 每个头的维度
) {
    int tok = blockIdx.x;    // 当前 token 索引
    int head = blockIdx.y;   // 当前注意力头索引
    int i = threadIdx.x;     // 当前线程处理的维度索引

    if (tok < seq_len && head < n_heads && i < d / 2) {
        int pos = start_pos + tok;  // 当前 token 的绝对位置
        float freq = pos / powf(theta, (2 * i) / (float)d);  // 计算频率
        float sin_freq, cos_freq;
        sincosf(freq, &sin_freq, &cos_freq);  // 计算 sin 和 cos

        // 获取当前复数对
        int idx = tok * n_heads * d + head * d + i;
        float a = y[idx];
        float b = y[idx + d / 2];

        // 应用 RoPE
        y[idx] = a * cos_freq - b * sin_freq;
        y[idx + d / 2] = b * cos_freq + a * sin_freq;
    }
}
