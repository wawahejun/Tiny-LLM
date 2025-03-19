# Tiny-LLM 项目报告

## 项目概述

Tiny-LLM 是一个基于 Rust 实现的高性能大语言模型推理引擎，支持 Llama 系列模型。该项目结合了 CUDA 加速、混合精度推理、Web 服务和多会话管理等功能，提供了一个完整的 LLM 推理解决方案。

## 功能特点

### 1. 高性能推理引擎
- 基于 Rust 实现的高效推理引擎
- 支持 Llama 系列模型
- 使用 KV Cache 优化推理性能
- 支持 top-k 和 top-p 采样策略

### 2. 混合精度推理
- 模型参数使用 FP16 格式存储，减少内存占用
- 计算过程中根据需要转换为 FP32 格式，保证计算精度
- 支持参数精度转换
- 预留 FP16 计算接口
- 支持 FP16/FP32 之间的高效转换

### 3. CUDA 加速
- 使用 CUDA 内核加速关键计算操作
- 优化的矩阵乘法实现（matmul_transb）
- 高效的 RMS 归一化实现
- 优化的 RoPE（旋转位置编码）实现

### 4. Web 服务与用户界面
- 基于 Axum 框架的 Web 服务
- 美观的响应式前端界面
- 支持模型选择（聊天模型/故事模型）
- 实时聊天交互

### 5. 多会话管理
- 支持创建多个独立会话
- 会话历史记录保存与加载
- 会话删除功能

### 6. 模型实现的独特之处
- 实现了分组查询注意力（GQA）机制，支持不同的注意力头配置
- 使用泛型设计支持不同精度的模型参数
- 高效的 SwiGLU 激活函数实现
- 支持流式输出的聊天模式
- 灵活的采样策略，同时支持 top-k 和 top-p 采样
- 优化的注意力计算和因果掩码实现
- 高效的内存管理和缓冲区复用


*部分代码说明[项目报告.pdf](项目报告.pdf)*

## 项目结构

```
Tiny-LLM/
├── src/
│   ├── config.rs          # 模型配置相关代码
│   ├── kvcache.rs         # KV缓存实现
│   ├── main.rs            # 主程序入口和Web服务
│   ├── model.rs           # 模型定义和推理实现
│   ├── operators.rs       # 算子实现
│   ├── params.rs          # 模型参数加载
│   ├── tensor.rs          # 张量实现
│   └── cuda_kernels/      # CUDA内核实现
│       ├── matmul_transb_kernel.cu  # 矩阵乘法内核
│       ├── rms_norm_kernel.cu       # RMS归一化内核
│       ├── rope_kernel.cu           # RoPE位置编码内核
│       └── compile_kernels.sh       # 内核编译脚本
├── static/
│   └── index.html         # Web前端界面
│  
├── models/
│   ├── chat/              # 聊天模型
│   └── story/             # 故事模型
├── build.rs               # 构建脚本
└── Cargo.toml             # 项目配置
```

## 使用方式

### 1. 环境准备

- 安装 Rust 和 Cargo
- 安装 CUDA 工具包（需要支持 CUDA 的 NVIDIA GPU）
- 准备 Llama 模型文件（safetensors 格式）

### 2. 编译项目

```bash
# 克隆项目
git clone https://github.com/your-username/Tiny-LLM.git
cd Tiny-LLM

# 编译CUDA内核
cd src/cuda_kernels
bash compile_kernels.sh


# 编译项目
cargo build --release
```

### 3. 模型准备

将模型文件放置在 `models` 目录下，结构如下：

```
models/
├── chat/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
└── story/
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

### 4. 启动服务

```bash
cargo run --release
```

服务默认在 `http://localhost:3000` 启动。

### 5. Web 界面使用

1. 打开浏览器访问 `http://localhost:3000`
2. 在界面上选择模型类型（聊天模型或故事模型）
3. 开始新会话或选择已有会话
4. 在输入框中输入消息并发送
5. 查看 AI 回复

### 6. API 使用

#### 发送聊天请求

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，请介绍一下自己",
    "model": "chat",
    "session_id": "optional_session_id"
  }'
```

#### 获取会话列表

```bash
curl http://localhost:3000/api/sessions
```

#### 获取特定会话历史

```bash
curl http://localhost:3000/api/sessions/session_id
```

#### 删除会话

```bash
curl -X DELETE http://localhost:3000/api/sessions/session_id
```

## 未来展望

1. **完整的混合精度支持**：
   - 实现 CUDA FP16 计算内核
   - 添加 INT8/INT4 量化支持
   - 实现自动混合精度训练（AMP）
   - 支持 BFloat16 格式

2. **性能优化**：
   - 实现 Flash Attention 机制
   - 优化 CUDA 内核性能
   - 添加 CPU 后端支持
   - 实现算子融合
   - 优化内存管理和显存使用

3. **功能增强**：
   - 实现流式输出
   - 添加长文本支持
   - 实现模型并行
   - 支持多 GPU 推理
   - 添加模型压缩功能

4. **架构升级**：
   - 支持更多模型架构（Mistral、Phi、Qwen等）
   - 实现模块化的模型结构
   - 添加模型转换工具
   - 支持动态批处理

5. **多模态支持**：
   - 添加图像处理能力
   - 支持语音输入输出
   - 实现跨模态理解
   - 添加文档理解功能

6. **开发体验**：
   - 完善错误处理机制
   - 添加详细的日志系统
   - 实现性能分析工具
   - 提供更多示例代码

7. **部署优化**：
   - 添加容器化支持
   - 实现分布式部署
   - 提供云端部署方案
   - 添加服务监控功能

8. **安全性增强**：
   - 实现用户认证系统
   - 添加内容过滤
   - 支持隐私数据保护
   - 实现访问控制

9. **工具链完善**：
   - 添加模型评估工具
   - 实现自动化测试
   - 提供性能基准测试
   - 添加调试工具

10. **生态系统集成**：
    - 支持主流深度学习框架
    - 添加常用 API 集成
    - 实现插件系统
    - 提供更多预训练模型支持

## 结论

Tiny-LLM 项目提供了一个高性能、易用的大语言模型推理引擎，通过混合精度推理、CUDA 加速、Web 服务和多会话管理等功能。该项目不仅可以作为简单实用的 LLM 推理工具，也是学习 Rust、CUDA 编程和大语言模型实现的参考。