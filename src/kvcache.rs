use std::vec;
use crate::tensor::Tensor;

pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    max_seq_len: usize,      // 最大序列长度
    dim: usize,              // 每层的维度
    length: usize,           // 当前缓存长度
}

impl<T: Default + Copy> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len,
            dim,
            length: init_len,
        }
    }

    /// 获取指定层的 Key 缓存切片
    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// 获取指定层的 Value 缓存切片
    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    /// 增加缓存长度
    pub fn increment(&mut self, seq_len: usize) {
        println!("Before increment: length = {}, seq_len = {}", self.length, seq_len);
        if self.length + seq_len > self.max_seq_len {
            // 如果缓存长度超过限制，丢弃最早的 token
            let overflow = (self.length + seq_len) - self.max_seq_len;
            println!("Overflow detected: overflow = {}", overflow);
            self.length = self.max_seq_len; // 更新缓存长度为最大长度
            for layer in 0..self.k_cache.len() {
                // 丢弃最早的 overflow 个 token
                self.k_cache[layer] = self.k_cache[layer].slice(overflow * self.dim, &vec![self.length - overflow, self.dim]);
                self.v_cache[layer] = self.v_cache[layer].slice(overflow * self.dim, &vec![self.length - overflow, self.dim]);
            }
        } else {
            self.length += seq_len;
        }
        println!("After increment: length = {}", self.length);
    }
    
    

    /// 获取当前缓存长度
    pub fn len(&self) -> usize {
        self.length
    }
}
