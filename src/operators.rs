use crate::tensor::Tensor;
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use std::ffi::CString;
use std::sync::{Arc, Once};
use std::cell::RefCell;
use std::thread_local;

// 用于 CUDA 初始化
static CUDA_INIT: Once = Once::new();

// 线程本地存储上下文
thread_local! {
    static CUDA_CONTEXT: RefCell<Option<Context>> = RefCell::new(None);
}

// 一次性初始化 CUDA
fn init_cuda() {
    CUDA_INIT.call_once(|| {
        // 初始化 CUDA
        rustacuda::init(CudaFlags::empty()).unwrap();
    });
    
    // 使用线程本地存储来管理上下文
    CUDA_CONTEXT.with(|ctx| {
        if ctx.borrow().is_none() {
            // 为当前线程创建新的上下文
            let device = Device::get_device(0).unwrap();
            let context = Context::create_and_push(
                ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, 
                device
            ).unwrap();
            
            // 存储上下文到线程本地存储
            *ctx.borrow_mut() = Some(context);
        }
        // 移除了尝试重新激活上下文的代码，因为 create_and_push 已经确保了上下文是活动的
    });
}

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    // Initialize CUDA if not already done
    init_cuda();
    
    // 加载 PTX 模块
    let ptx = match CString::new(include_str!("../rope_kernel.ptx")) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("创建 CString 失败: {:?}", e);
            return;
        }
    };
    
    let module = match Module::load_from_string(&ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("加载 PTX 模块失败: {:?}", e);
            return;
        }
    };

    // 获取内核函数
    let func_name = CString::new("rope").unwrap();
    let function = match module.get_function(&func_name) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("获取函数 'rope' 失败: {:?}", e);
            return;
        }
    };

    // 获取张量形状
    let shape = y.shape();
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    
    // 使用 y 的数据指针而不是整个 Tensor
    let mut data_gpu = unsafe { DeviceBuffer::from_slice(y.data_mut()).unwrap() };
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

    // 启动内核
    let grid_size = (seq_len as u32, n_heads as u32, 1);
    let block_size = BlockSize::x(d as u32 / 2); // 每个线程处理一个复数对
    
    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            data_gpu.as_device_ptr(),
            start_pos as i32,
            theta,
            seq_len as i32,
            n_heads as i32,
            d as i32
        )).unwrap();
    }

    // 将结果复制回 CPU
    data_gpu.copy_to(unsafe { y.data_mut() }).unwrap();
    
    // 确保 GPU 操作完成
    stream.synchronize().unwrap();
}

pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };

    // 修改：添加数值稳定性处理
    let epsilon = 1e-6f32;

    for b in 0..batch {
        let base = b * seq_len * total_seq_len;

        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            // 修改：处理可能的NaN或inf值
            let max = data[offset..offset + boundary]
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| {
                    if b.is_finite() { a.max(b) } else { a }
                });

            let sum = (0..boundary)
                .map(|j| {
                    let e = if data[offset + j].is_finite() {
                        (data[offset + j] - max).exp()
                    } else {
                        0.0
                    };
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            // 修改：添加数值稳定性检查
            if sum > epsilon {
                (0..boundary).for_each(|j| data[offset + j] /= sum);
            } else {
                // 如果和太小，使用均匀分布
                let uniform_value = 1.0 / boundary as f32;
                (0..boundary).for_each(|j| data[offset + j] = uniform_value);
            }

            // 将剩余值设置为0.0
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // Initialize CUDA if not already done
    init_cuda();
    
    // 加载 PTX 模块
    let ptx = CString::new(include_str!("../rms_norm_kernel.ptx")).unwrap();
    let module = match Module::load_from_string(&ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("加载 PTX 模块失败: {:?}", e);
            return;
        }
    };

    // 获取内核函数
    let func_name = CString::new("rms_norm").unwrap();
    let function = module.get_function(&func_name).unwrap();

    // 使用 DeviceBuffer 而不是 DeviceBox
    let mut x_gpu = unsafe { DeviceBuffer::from_slice(x.data()).unwrap() };
    let mut w_gpu = unsafe { DeviceBuffer::from_slice(w.data()).unwrap() };
    let mut y_gpu = unsafe { DeviceBuffer::from_slice(y.data_mut()).unwrap() };
    
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

    // 启动内核
    let grid_size = (y.size() as u32 / w.size() as u32, 1, 1);
    let block_size = BlockSize::x(w.size() as u32);
    
    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            x_gpu.as_device_ptr(),
            w_gpu.as_device_ptr(),
            y_gpu.as_device_ptr(),
            epsilon,
            w.size() as i32  // 添加缺少的 dim 参数
        )).unwrap();
    }

    // 将结果复制回 CPU
    y_gpu.copy_to(unsafe { y.data_mut() }).unwrap();
    
    // 确保 GPU 操作完成
    stream.synchronize().unwrap();
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();
    for i in 0..len {
        y_data[i] *= x_data[i] / (1.0 + (-x_data[i]).exp());
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // Initialize CUDA if not already done
    init_cuda();

    // 加载 PTX 模块
    let ptx = CString::new(include_str!("../matmul_kernel.ptx")).unwrap();
    let module = match Module::load_from_string(&ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("加载 PTX 模块失败: {:?}", e);
            return;
        }
    };

    // 获取内核函数
    let func_name = CString::new("matmul_transb").unwrap();
    let function = module.get_function(&func_name).unwrap();

    // 使用 DeviceBuffer 而不是 DeviceBox
    let mut a_gpu = unsafe { DeviceBuffer::from_slice(a.data()).unwrap() };
    let mut b_gpu = unsafe { DeviceBuffer::from_slice(b.data()).unwrap() };
    let mut c_gpu = unsafe { DeviceBuffer::from_slice(c.data_mut()).unwrap() };
    
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

    // 启动内核
    let grid_size = (c.shape()[0] as u32, c.shape()[1] as u32, 1);
    let block_size = BlockSize::xy(16, 16);
    
    unsafe {
        launch!(function<<<grid_size, block_size, 0, stream>>>(
            a_gpu.as_device_ptr(),
            b_gpu.as_device_ptr(),
            c_gpu.as_device_ptr(),
            beta,
            alpha,
            c.shape()[0] as i32,
            c.shape()[1] as i32,
            a.shape()[1] as i32
        )).unwrap();
    }

    // 将结果复制回 CPU
    c_gpu.copy_to(unsafe { c.data_mut() }).unwrap();
    
    // 确保 GPU 操作完成
    stream.synchronize().unwrap();
}


// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}