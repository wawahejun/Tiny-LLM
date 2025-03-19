use std::process::Command;

fn main() {
    // 编译 CUDA 内核
    let status = Command::new("sh")
        .arg("src/cuda_kernels/compile_kernels.sh")
        .status()
        .expect("Failed to compile CUDA kernels");

    if !status.success() {
        panic!("CUDA kernel compilation failed");
    }

    // 告诉 Cargo 如果 CUDA 内核文件发生变化，需要重新构建
    println!("cargo:rerun-if-changed=src/cuda_kernels/matmul_transb_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda_kernels/rms_norm_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda_kernels/rope_kernel.cu");
}
