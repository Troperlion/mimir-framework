// available_count() et dtypes_count() depuis env vars (pas de JSON parse).
let ac = Architectures::available_count();
println!("[bridge-rs] available count: {}", ac);

let dc = Architectures::dtypes_count();
println!("[bridge-rs] dtypes count: {}", dc);

// dtypes() retourne JSON normalise: aliases est un array, pas une string CSV.
let dtypes_json = Architectures::dtypes();
println!("[bridge-rs] dtypes json has aliases array: {}", dtypes_json.contains("[\"float\"") || dtypes_json.contains("[\\\"float\\\""));

Model::create("basic_mlp", "");
Model::allocate_params();
Model::init_weights("he", 42);
let p = Model::total_params();
println!("[bridge-rs] total_params (cached): {}", p);
println!("[bridge-rs] ok");
