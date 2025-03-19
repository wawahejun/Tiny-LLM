use core::slice;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
use half::f16;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f16> {
    #[allow(unstable_features)]
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 辅助函数：安全地获取张量
        let get_tensor = |name: &str| -> Tensor<f16> {
            match safetensor.tensor(name) {
                Ok(data) => {
                    let p: usize = data.shape().iter().product();
                    let new_data = unsafe {
                        std::slice::from_raw_parts(data.data().as_ptr() as *const f16, p)
                    };
                    Tensor::new(Vec::from(new_data), &data.shape().to_vec())
                }
                Err(_) => {
                    eprintln!("Warning: Failed to load tensor: {}", name);
                    Tensor::new(vec![f16::from_f32(0.0)], &vec![1])
                }
            }
        };

        // 辅助函数：获取各层的张量
        let get_layer_tensors = |prefix: &str, suffix: &str| -> Vec<Tensor<f16>> {
            (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("{}.{}.{}", prefix, i, suffix)))
                .collect()
        };

        let embedding_table = if config.tie_word_embeddings {
            get_tensor("lm_head.weight")
        } else {
            get_tensor("model.embed_tokens.weight")
        };

        LLamaParams {
            embedding_table: embedding_table,
            rms_att_w: get_layer_tensors("model.layers", "input_layernorm.weight"),
            wq: get_layer_tensors("model.layers", "self_attn.q_proj.weight"),
            wk: get_layer_tensors("model.layers", "self_attn.k_proj.weight"),
            wv: get_layer_tensors("model.layers", "self_attn.v_proj.weight"),
            wo: get_layer_tensors("model.layers", "self_attn.o_proj.weight"),
            rms_ffn_w: get_layer_tensors("model.layers", "post_attention_layernorm.weight"),
            w_up: get_layer_tensors("model.layers", "mlp.up_proj.weight"),
            w_gate: get_layer_tensors("model.layers", "mlp.gate_proj.weight"),
            w_down: get_layer_tensors("model.layers", "mlp.down_proj.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}