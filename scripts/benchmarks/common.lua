-- Helpers partagés des benchmarks natifs Mímir.
local B = {}
B.results = {}
B.started_at = os.date('!%Y-%m-%dT%H:%M:%SZ')
function B.has(flag)
  for _, v in ipairs(arg or {}) do if v == flag then return true end end
  return false
end
function B.value(flag, env, default)
  for i, v in ipairs(arg or {}) do
    if v == flag then
      assert(arg[i + 1] and not arg[i + 1]:match('^%-%-'), 'valeur manquante: ' .. flag)
      return arg[i + 1]
    end
  end
  return (env and os.getenv(env)) or default
end
function B.number(flag, env, default, minimum, integer)
  local n = tonumber(B.value(flag, env, default))
  assert(n and n == n and n < math.huge and n >= (minimum or 0)
    and (not integer or n % 1 == 0), 'valeur invalide: ' .. flag)
  return n
end
function B.finite(v)
  return type(v) == 'number' and v == v and math.abs(v) < math.huge
end
function B.vector(v, expected)
  assert(type(v) == 'table' and #v > 0, 'sortie vide ou invalide')
  assert(not expected or #v == expected,
    'taille de sortie inattendue: ' .. #v .. ' (attendue: ' .. tostring(expected) .. ')')
  for i, x in ipairs(v) do assert(B.finite(x), 'NaN/Inf à l’indice ' .. i) end
  return v
end
function B.floats(n)
  local x = {}
  for i = 1, n do x[i] = math.sin(i * 0.73) * 0.25 end
  return x
end
function B.tokens(n, vocab)
  local x = {}
  for i = 1, n do x[i] = 1 + (i % (vocab - 1)) end
  return x
end
function B.transformer_input(n, vocab, dim)
  local zeros = {}
  for i = 1, n * dim do zeros[i] = 0 end
  return {__input__=B.tokens(n, vocab), mag=zeros, mod=zeros}
end
assert(Mimir.Benchmark and type(Mimir.Benchmark.now) == 'function',
  'Mimir.Benchmark.now indisponible: reconstruire le binaire Mimir')
B.now = Mimir.Benchmark.now
function B.measure(label, fn)
  local start = B.now()
  local result = fn()
  local elapsed = B.now() - start
  log(string.format('%s: wall_ms=%.3f', label, elapsed * 1000))
  B.results[#B.results + 1] = {name=label, kind='operation', samples_ms={elapsed * 1000}}
  return result, elapsed
end

local function percentile(sorted, ratio)
  if #sorted == 0 then return 0 end
  local index = math.max(1, math.ceil(#sorted * ratio))
  return sorted[math.min(index, #sorted)]
end

local function stats(samples)
  local sorted, sum = {}, 0
  for i, value in ipairs(samples) do sorted[i], sum = value, sum + value end
  table.sort(sorted)
  return {
    min_ms=sorted[1] or 0,
    median_ms=percentile(sorted, 0.5),
    p95_ms=percentile(sorted, 0.95),
    max_ms=sorted[#sorted] or 0,
    mean_ms=#sorted > 0 and sum / #sorted or 0,
  }
end

function B.record(name, samples, fields)
  assert(type(samples) == 'table' and #samples > 0, 'échantillons manquants: ' .. name)
  local result = fields or {}
  result.name = name
  result.kind = result.kind or 'benchmark'
  result.samples_ms = samples
  result.stats = stats(samples)
  result.iterations_per_second = result.stats.median_ms > 0
    and 1000 / result.stats.median_ms or math.huge
  B.results[#B.results + 1] = result
  return result
end
function B.setup(default_ram)
  local ram = B.number('--ram', 'MIMIR_BENCH_RAM_GB', default_ram or 2, 0.001)
  assert(ram <= 1000, '--ram doit être exprimé en GiB (<=1000)')
  assert(Mimir.MemoryGuard.setLimit(ram))
  assert(Mimir.Allocator.configure({max_ram_gb=ram,
    enable_compression=not B.has('--no-compress') and os.getenv('MIMIR_BENCH_COMPRESS') ~= '0'
      and os.getenv('MIMIR_BENCH_COMPRESS') ~= 'false'}))
  assert(Mimir.Model.set_hardware(not B.has('--no-accel')))
  local caps = Mimir.Model.hardware_caps()
  log(string.format('Chronométrage: murale monotone (steady_clock), OMP_NUM_THREADS=%s, AVX2=%s, FMA=%s',
    os.getenv('OMP_NUM_THREADS') or 'auto', tostring(caps.avx2), tostring(caps.fma)))
  log('set_hardware contrôle les kernels accélérés; ce booléen ne sélectionne pas un backend GPU.')
end
function B.dtype()
  assert(Mimir.Model.dtype(B.value('--dtype', 'MIMIR_DTYPE', 'float32')))
end
function B.init()
  B.dtype()
  B.measure('allocate_params', function() assert(Mimir.Model.allocate_params()) end)
  B.measure('init_weights', function() assert(Mimir.Model.init_weights('xavier', 42)) end)
  log('params=' .. Mimir.Model.total_params() .. ' dtype=' .. Mimir.Model.dtype())
end
function B.create(name, cfg)
  B.measure('create ' .. name .. ' (graphe inclus)', function()
    assert(Mimir.Model.create(name, cfg))
  end)
  B.current_model = name
  B.init()
end
function B.forward(input, training, expected)
  return B.vector(assert(Mimir.Model.forward(input, training or false)), expected)
end
function B.bench_forward(input, expected)
  local warmup = B.number('--warmup', nil, 1, 0, true)
  local iters = B.number('--iters', 'MIMIR_BENCH_ITERS', B.has('--quick') and 2 or 10, 1, true)
  for _ = 1, warmup do B.forward(input, false, expected) end
  local output, samples = nil, {}
  for i = 1, iters do
    local started = B.now()
    output = assert(Mimir.Model.forward(input, false))
    samples[i] = (B.now() - started) * 1000
  end
  B.vector(output, expected)
  local result = B.record('forward ' .. (B.current_model or 'model'), samples, {
    warmup=warmup, iterations=iters, output_size=#output,
  })
  local summary, throughput = result.stats, result.iterations_per_second
  log(string.format(
    'forward: warmup=%d iters=%d output=%d | median=%.3f ms p95=%.3f ms min=%.3f ms max=%.3f ms | %.2f iter/s',
    warmup, iters, #output, summary.median_ms, summary.p95_ms,
    summary.min_ms, summary.max_ms, throughput))
  return output
end
function B.snapshot(tag)
  local g, a = Mimir.MemoryGuard.getStats(), Mimir.Allocator.getStats()
  log(string.format('%s: guard_current_MiB=%.2f guard_peak_MiB=%.2f tensors=%d loaded=%d',
    tag, g.current_mb, g.peak_mb, a.tensor_count, a.loaded_count))
  -- Compteurs internes, pas le RSS; pic cumulatif, aucun reset avec modèle vivant.
end
function B.transformer(dim, layers, seq, vocab, causal)
  return {d_model=dim, num_layers=layers, num_heads=4, mlp_hidden=dim*4,
    output_dim=dim, seq_len=seq, vocab_size=vocab, padding_idx=0, causal=causal or false}
end
function B.roundtrip(cfg, input)
  B.create('basic_mlp', cfg)
  local before = B.forward(input)
  local path = os.tmpname()
  local ok, err = pcall(function()
    B.measure('save safetensors', function()
      assert(Mimir.Serialization.save(path, 'safetensors', {save_tokenizer=false, save_encoder=false}))
    end)
    B.create('basic_mlp', cfg)
    assert(Mimir.Model.init_weights('zeros', 0))
    local reset = B.forward(input, false, #before)
    local changed = false
    for i, v in ipairs(before) do if math.abs(v-reset[i]) > 1e-5 then changed=true end end
    assert(changed, 'roundtrip non discriminant: poids inchangés')
    B.measure('load safetensors', function()
      assert(Mimir.Serialization.load(path, 'safetensors', {load_tokenizer=false, load_encoder=false, strict_mode=true}))
    end)
    local after = B.forward(input, false, #before)
    for i, v in ipairs(before) do assert(math.abs(v-after[i]) < 1e-5, 'roundtrip différent') end
  end)
  os.remove(path)
  assert(ok, err)
end

local function json_escape(value)
  return tostring(value):gsub('\\', '\\\\'):gsub('"', '\\"')
    :gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
end

local function json_value(value)
  local kind = type(value)
  if kind == 'number' then
    return B.finite(value) and string.format('%.9g', value) or 'null'
  elseif kind == 'boolean' then
    return tostring(value)
  elseif kind == 'string' then
    return '"' .. json_escape(value) .. '"'
  elseif kind ~= 'table' then
    return 'null'
  end
  local array = #value > 0
  local parts = {}
  if array then
    for i = 1, #value do parts[i] = json_value(value[i]) end
    return '[' .. table.concat(parts, ',') .. ']'
  end
  for key, item in pairs(value) do
    parts[#parts + 1] = '"' .. json_escape(key) .. '":' .. json_value(item)
  end
  table.sort(parts)
  return '{' .. table.concat(parts, ',') .. '}'
end

function B.report(name)
  local path = B.value('--report', 'MIMIR_BENCH_REPORT', nil)
  if not path or path == '' then return end
  local report = {
    benchmark=name,
    started_at=B.started_at,
    finished_at=os.date('!%Y-%m-%dT%H:%M:%SZ'),
    clock='steady_clock',
    omp_threads=os.getenv('OMP_NUM_THREADS') or 'auto',
    results=B.results,
  }
  local file, err = io.open(path, 'w')
  assert(file, 'rapport impossible: ' .. tostring(err))
  if path:match('%.md$') then
    file:write('# Benchmark ', name, '\n\n')
    file:write('| Mesure | Type | Médiane murale | p95 murale | Itérations/s |\n')
    file:write('|---|---:|---:|---:|---:|\n')
    for _, result in ipairs(B.results) do
      local summary = result.stats or stats(result.samples_ms or {})
      file:write(string.format('| %s | %s | %.3f ms | %.3f ms | %.2f |\n',
        result.name, result.kind, summary.median_ms, summary.p95_ms,
        result.iterations_per_second or 0))
    end
  else
    file:write(json_value(report), '\n')
  end
  file:close()
  log('rapport=' .. path)
end
return B
