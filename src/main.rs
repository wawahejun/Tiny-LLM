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


use axum::{
    routing::{get, post, delete},
    Router,
    Json,
    http::{StatusCode, HeaderMap},
    response::IntoResponse,
    extract::{State, Path},
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use tower_http::services::ServeDir;

#[derive(Clone)]
struct AppState {
    sessions: Arc<Mutex<HashMap<String, Vec<(String, String)>>>>, // 会话 ID -> 历史记录
    next_session_id: Arc<Mutex<u32>>, // 用于生成顺序编号
}

impl AppState {
    fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            next_session_id: Arc::new(Mutex::new(1)), // 从 1 开始
        }
    }

    fn generate_session_id(&self) -> String {
        let mut next_id = self.next_session_id.lock().unwrap();
        let session_id = format!("Session {}", *next_id);
        *next_id += 1;
        session_id
    }
}

#[derive(Deserialize, Serialize)]
struct ChatRequest {
    input: String,
    model: String,
    session_id: Option<String>, // 会话 ID（可选）
}

#[derive(Deserialize, Serialize)]
struct ChatResponse {
    response: String,
    session_id: String, // 返回会话 ID
}

#[derive(Deserialize, Serialize)]
struct SessionsResponse {
    sessions: Vec<String>, // 所有会话 ID
}

#[derive(Deserialize, Serialize)]
struct SessionHistoryResponse {
    history: Vec<(String, String)>, // 会话历史记录
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(payload): Json<ChatRequest>,
) -> impl IntoResponse {
    // 获取或创建会话 ID
    let session_id = payload.session_id.unwrap_or_else(|| state.generate_session_id());

    // 加载模型
    let model_dir = format!("models/{}", payload.model);
    let llama = model::Llama::from_safetensors(&model_dir);
    let mut cache = llama.new_cache();

    // 获取会话历史记录
    let mut sessions = state.sessions.lock().unwrap();
    let history = sessions.entry(session_id.clone()).or_insert_with(Vec::new);

    // 将历史记录转换为模型输入
    let messages: Vec<(&str, &str)> = history
        .iter()
        .map(|(user, ai)| ("user", user.as_str()))
        .chain(std::iter::once(("user", payload.input.as_str())))
        .collect();

    // 调用模型生成回复
    let response = llama.chat(&messages, &mut cache, 100, 0.6, 40, 0.6);

    // 更新会话历史记录
    history.push((payload.input, response.clone()));

    // 返回响应
    (StatusCode::OK, Json(ChatResponse { response, session_id }))
}

async fn list_sessions_handler(State(state): State<AppState>) -> impl IntoResponse {
    // 获取所有会话 ID
    let sessions = state.sessions.lock().unwrap();
    let session_ids: Vec<String> = sessions.keys().cloned().collect();
    (StatusCode::OK, Json(SessionsResponse { sessions: session_ids }))
}

async fn get_session_history_handler(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    // 获取指定会话的历史记录
    let sessions = state.sessions.lock().unwrap();
    if let Some(history) = sessions.get(&session_id) {
        (StatusCode::OK, Json(SessionHistoryResponse { history: history.clone() }))
    } else {
        // 如果会话不存在，返回空的历史记录
        (StatusCode::OK, Json(SessionHistoryResponse { history: Vec::new() }))
    }
}

async fn delete_session_handler(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    // 删除会话
    let mut sessions = state.sessions.lock().unwrap();
    sessions.remove(&session_id);
    (StatusCode::OK, Json(()))
}

async fn index_handler() -> impl IntoResponse {
    // 设置响应头为 text/html
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "text/html".parse().unwrap());

    // 返回前端页面
    (headers, include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    // 初始化会话管理
    let state = AppState::new();

    // 创建路由
    let app = Router::new()
        .route("/", get(index_handler)) // 提供前端页面
        .route("/api/chat", post(chat_handler)) // 提供聊天 API
        .route("/api/sessions", get(list_sessions_handler)) // 获取会话列表
        .route("/api/sessions/:session_id", get(get_session_history_handler)) // 获取会话历史记录
        .route("/api/sessions/:session_id", delete(delete_session_handler)) // 删除会话
        .nest_service("/static", ServeDir::new("static")) // 提供静态文件服务
        .with_state(state); // 添加状态管理

    // 启动服务器
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server running at http://{}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
