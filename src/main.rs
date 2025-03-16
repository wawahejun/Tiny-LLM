mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::env;
use model::Llama;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use serde::{Deserialize, Serialize};



fn main() {
    // 解析命令行参数
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_name>", args[0]);
        std::process::exit(1);
    }

    let model_name = &args[1]; // 获取模型名称（chat 或 story）
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models").join(model_name);

    println!("Loading model from: {:?}", model_dir);

    // 加载模型
    let llama = Llama::from_safetensors(&model_dir);
    let mut cache = llama.new_cache();

    loop {
        print!("You: ");
        std::io::stdout().flush().unwrap();
        let mut user_input = String::new();
        std::io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();

        if user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        let messages = vec![
            ("user", user_input),
        ];

        // 生成回复
        let response = llama.chat(&messages, &mut cache, 100, 0.9, 40, 0.7);
        println!("AI: {}", response);
    }
}
