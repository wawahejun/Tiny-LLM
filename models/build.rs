fn main() {
    // 检查CUDA工具链
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudnn");
        
        // 编译CUDA代码
        cc::Build::new()
            .cuda(true)
            .file("src/cuda_kernels.cu")
            .compile("cuda_kernels");
    }
}