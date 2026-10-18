// Minimal Rust snippet for --rust runtime (injected inside generated main()).
println!("[rust] mimir script started");
let arg_json = std::env::var("MIMIR_ARG_JSON").unwrap_or_else(|_| "[]".to_string());
let conf_path = std::env::var("MIMIR_CONF_PATH").unwrap_or_default();
let conf_dir = std::env::var("MIMIR_CONF_DIR").unwrap_or_default();
println!("[rust] arg json: {}", arg_json);
println!("[rust] conf path: {}", conf_path);
println!("[rust] conf dir: {}", conf_dir);
println!("[rust] done");
