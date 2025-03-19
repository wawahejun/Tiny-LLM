use std::cmp::{min, Ordering};
use std::fs::File;
use std::vec;
use half::f16; 

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
use crate::operators::*;
use tokenizers::Tokenizer;
use std::path::PathBuf;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q  // Q头数量是KV头的整数倍，所以nqh = nkvh * n_groups
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
    model_dir: PathBuf, 
}

fn easy_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = *logits.iter().reduce(|a, b| if a > b { a } else { b }).unwrap();
    let exp_logits: Vec<_> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();
    let result: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp_logits).collect();
    result.to_vec()
}

macro_rules! flush_print {
    ($fmt:expr) => {{
        use std::io;
        use std::io::Write;
        let mut stdout = io::stdout();
        write!(stdout, $fmt).unwrap();
        stdout.flush().unwrap();
    }};
    ($fmt:expr, $($arg:tt)*) => {{
        use std::io;
        use std::io::Write;
        let mut stdout = io::stdout();
        write!(stdout, $fmt, $($arg)*).unwrap();
        stdout.flush().unwrap();
    }};
}

impl Llama<f16> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();

        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
            model_dir: model_dir.as_ref().to_path_buf(),
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table.to_f32());
        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer].to_f32(),
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let mut k = cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let mut v = cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer].to_f32(), 1.0);
            OP::matmul_transb(&mut k, 0., &hidden_states, &self.params.wk[layer].to_f32(), 1.0);
            OP::matmul_transb(&mut v, 0., &hidden_states, &self.params.wv[layer].to_f32(), 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                &full_k,
                &full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            matmul_transb(&mut residual, 1f32, &hidden_states, &self.params.wo[layer].to_f32(), 1f32);

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer].to_f32(),
                &self.params.w_down[layer].to_f32(),
                &self.params.w_gate[layer].to_f32(),
                &self.params.rms_ffn_w[layer].to_f32(),
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w.to_f32(),
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head.to_f32(), 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::from(token_ids);
        result.push(self.bos_token_id);

        let mut input = result.clone();
        let mut cache = KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0);

        while result.len() < max_len {
            let mut logits = self.forward(&Tensor::<u32>::new(input.clone(), &vec![input.len()]), &mut cache);
            let length = logits.size();
            let data = unsafe { logits.data_mut() };
            if temperature > 0. {
                for i in 0..length {
                    data[i] /= temperature;
                }
            }
            let logits = easy_softmax(logits.data());
            let new_word_id = Self::select_word_to_id(&logits.to_vec(), top_p, top_k as usize);
            if new_word_id == self.eos_token_id {
                break;
            }
            result.push(new_word_id);
            input = vec![new_word_id];

            // 逐步输出
            use std::path::PathBuf;
            use tokenizers::Tokenizer;
            let project_dir = env!("CARGO_MANIFEST_DIR");
            let model_dir = PathBuf::from(project_dir).join("models").join("story");
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            flush_print!("{}", tokenizer.decode(&[new_word_id], true).unwrap());
        }

        if result.len() == max_len {
            println!(" <|heke: is max len. its maybe has some error|>");
        }

        result
    }

    pub fn chat(
        &self,
        messages: &[(&str, &str)], // (role, content) pairs
        cache: &mut KVCache<f32>,
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> String {
        // Build the prompt using the Jinja2 template
        let mut prompt = String::new();
        for (role, content) in messages {
            prompt.push_str(&format!("<|im_start|>{}", role));
            prompt.push('\n');
            prompt.push_str(content);
            prompt.push_str("<|im_end|>");
            prompt.push('\n');
        }
        prompt.push_str("<|im_start|>assistant\n");

        // Tokenize the prompt
        let tokenizer = self.load_tokenizer();
        let input_ids = tokenizer.encode(&*prompt, true).unwrap().get_ids().to_vec();

        // Generate the response
        let mut result = Vec::new();
        let mut input = input_ids.clone();
        while result.len() < max_len {
            let mut logits = self.forward(&Tensor::<u32>::new(input.clone(), &vec![input.len()]), cache);
            let length = logits.size();
            let data = unsafe { logits.data_mut() };
            if temperature > 0. {
                for i in 0..length {
                    data[i] /= temperature;
                }
            }
            let logits = easy_softmax(logits.data());
            let new_word_id = Self::select_word_to_id(&logits.to_vec(), top_p, top_k as usize);
            if new_word_id == self.eos_token_id {
                break;
            }
            result.push(new_word_id);
            input = vec![new_word_id];
        }

        // Decode the response
        tokenizer.decode(&result, true).unwrap()
    }

    fn load_tokenizer(&self) -> Tokenizer {
        use std::path::PathBuf;
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models").join("story");

        Tokenizer::from_file(self.model_dir.join("tokenizer.json"))
        .expect("Failed to load tokenizer. Please ensure tokenizer.json exists in the correct path.")
    }
    
    fn select_word_to_id(logits: &Vec<f32>, top_p: f32, top_k: usize) -> u32 {
        let mut indices_and_values: Vec<(_, _)> = logits
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, value))
            .collect();

        indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut take_num = 1; // take_num 最小是1
        let mut tmp_sump = 0.;

        if top_p > 0. {
            for i in indices_and_values.iter() {
                tmp_sump += i.1;
                if tmp_sump > top_p {
                    break;
                }
                take_num += 1;
            }
        }

        take_num = if top_k > 0 { min(take_num, top_k) } else { take_num };

        let top_indices: Vec<f32> = indices_and_values
            .iter()
            .cloned()
            .take(take_num)
            .map(|(_, w)| w)
            .collect();

        let resampled = easy_softmax(&*top_indices);

        use rand::distributions::Distribution;
        use rand::distributions::WeightedIndex;

        let mut rng = rand::thread_rng();
        let index = WeightedIndex::new(resampled)
            .unwrap()
            .sample(&mut rng);

        indices_and_values[index].0 as u32
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>,
    att_scores: &mut Tensor<f32>,
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    let bias: usize = seq_len * total_seq_len;
    let scale = 1.0 / ((dqkv as f32).sqrt());
    
    // 计算注意力分数
    for head in 0..n_kv_h {
        for group in 0..n_groups {
            let bh = n_groups * head + group;
            let (_q, _k, _scores) = (q.data(), k.data(), unsafe { att_scores.data_mut() });
            
            for mi in 0..seq_len {
                let q_base = mi * n_kv_h * n_groups * dqkv + bh * dqkv;
                for ni in 0..total_seq_len {
                    let k_base = ni * n_kv_h * dqkv + head * dqkv;
                    let idx = bh * bias + mi * total_seq_len + ni;
                    
                    // 计算注意力分数
                    let mut score = 0.0f32;
                    for ki in 0..dqkv {
                        score += _q[q_base + ki] * _k[k_base + ki];
                    }
                    _scores[idx] = score * scale;
                    
                    // 应用因果掩码
                    if ni >= (total_seq_len - seq_len + mi) {
                        _scores[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    // 应用 softmax
    masked_softmax(att_scores);

    // 计算输出
    let (_attn, _v, _out) = (att_scores.data(), v.data(), unsafe { hidden_states.data_mut() });
    for head in 0..n_kv_h {
        for group in 0..n_groups {
            let bh = n_groups * head + group;
            for i in 0..seq_len {
                let out_base = i * n_kv_h * n_groups * dqkv + bh * dqkv;
                let attn_base = bh * bias + i * total_seq_len;
                
                for j in 0..dqkv {
                    let mut sum = 0.0f32;
                    for t in 0..total_seq_len {
                        sum += _attn[attn_base + t] * _v[t * n_kv_h * dqkv + head * dqkv + j];
                    }
                    _out[out_base + j] = sum;
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,     // x 以及之后的 y
    hidden_states: &mut Tensor<f32>,    // 用于存储过程中的计算结果
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0f32, hidden_states, w_gate, 1f32);
    matmul_transb(up, 0f32, hidden_states, w_up, 1f32);
    swiglu(up, gate);
    matmul_transb(residual, 1f32, up, w_down, 1f32);
}
