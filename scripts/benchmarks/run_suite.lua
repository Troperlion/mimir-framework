---@diagnostic disable: undefined-global

local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local function has(flag)
  for _, value in ipairs(arg or {}) do
    if value == flag then return true end
  end
  return false
end

local function shell_quote(value)
  return "'" .. tostring(value):gsub("'", "'\\''") .. "'"
end

local function project_root()
  assert(type(ROOTWORK) == "string" and ROOTWORK ~= "", "ROOTWORK absent")
  return ROOTWORK
end

local function option_value(flag, default)
  for index, item in ipairs(arg or {}) do
    if item == flag then
      assert(arg[index + 1] and not arg[index + 1]:match("^%-%-"), flag .. ": valeur manquante")
      return arg[index + 1]
    end
  end
  return default
end

local suite = "core"
for i, value in ipairs(arg or {}) do
  if value == "--suite" then
    assert(arg[i+1] and not arg[i+1]:match("^%-%-"), "--suite: valeur manquante")
  elseif value:match("^%-%-") then
    assert(value == "--" or value == "--keep-going" or value == "--quick"
      or value == "--report-dir" or value == "--unique-run-dir",
      "option inconnue: " .. value)
  end
end
for i = 1, #(arg or {}) do
  if arg[i] == "--suite" and arg[i + 1] then suite = arg[i + 1] end
end
if has("--quick") then
  assert(suite == "core" or suite == "quick",
    "--quick est un alias de --suite quick et ne se combine pas avec une autre suite")
  suite = "quick"
end

local groups = {
  smoke = {
    {"dtype API", "scripts/benchmarks/dtype_api_smoke.lua"},
    {"spill cleanup", "scripts/benchmarks/spill_cleanup_smoke.lua"},
    {"NMS", "scripts/benchmarks/benchmark_nms.lua", "--quick"},
  },
  core = {
    {"NMS", "scripts/benchmarks/benchmark_nms.lua"},
    {"attention", "scripts/benchmarks/benchmark_attention.lua"},
    {"convolution training", "scripts/benchmarks/benchmark_conv_train.lua"},
    {"général", "scripts/benchmarks/benchmark.lua"},
    {"complet", "scripts/benchmarks/benchmark_complet.lua"},
    {"officiel", "scripts/benchmarks/benchmark_official.lua", "--safe", "--iters", "5"},
  },
  quick = {
    {"NMS", "scripts/benchmarks/benchmark_nms.lua", "--quick"},
    {"attention", "scripts/benchmarks/benchmark_attention.lua", "--quick", "--iters", "2", "--warmup", "1"},
    {"convolution training", "scripts/benchmarks/benchmark_conv_train.lua", "--quick"},
    {"général", "scripts/benchmarks/benchmark.lua", "--quick", "--iters", "2", "--warmup", "1"},
    {"complet", "scripts/benchmarks/benchmark_complet.lua", "--quick", "--iters", "2", "--warmup", "1"},
    {"officiel", "scripts/benchmarks/benchmark_official.lua", "--quick", "--iters", "1"},
  },
  stress = {
    {"stress", "scripts/benchmarks/benchmark_stress.lua"},
  },
}

groups.all = {
  groups.smoke[1],
  groups.smoke[2],
  groups.core[1],
  groups.core[2],
  groups.core[3],
  groups.core[4],
  groups.core[5],
  groups.core[6],
  groups.stress[1],
}

local selected = groups[suite]
assert(selected, "suite inconnue: " .. tostring(suite))

local root = project_root()
local mimir_bin = os.getenv("MIMIR_BIN") or (root .. "/bin/mimir")
local requested_report_dir = option_value(
  "--report-dir", os.getenv("MIMIR_BENCH_REPORT_DIR") or "outputs/benchmarks/latest")
local report_dir = FS.normalize(requested_report_dir)
local is_absolute = FS.is_windows()
  and (report_dir:match("^%a:[/\\]") ~= nil or report_dir:match("^[/\\][/\\]") ~= nil)
  or report_dir:sub(1, 1) == "/"
if not is_absolute then report_dir = FS.join(root, report_dir) end
if has("--unique-run-dir") then
  local run_id = os.date("!%Y%m%dT%H%M%SZ")
  local candidate = FS.join(report_dir, run_id)
  local suffix = 1
  while FS.exists(candidate) do
    candidate = FS.join(report_dir, string.format("%s-%02d", run_id, suffix))
    suffix = suffix + 1
  end
  report_dir = candidate
end
assert(FS.mkdir_p(report_dir), "impossible de créer le dossier de rapports: " .. report_dir)
local failures = {}
local summary = {}

local function report_slug(name)
  local normalized = name:lower()
  for source, replacement in pairs({
    ["é"]="e", ["è"]="e", ["ê"]="e", ["ë"]="e",
    ["à"]="a", ["â"]="a", ["ä"]="a", ["î"]="i", ["ï"]="i",
    ["ô"]="o", ["ö"]="o", ["ù"]="u", ["û"]="u", ["ü"]="u", ["ç"]="c",
  }) do
    normalized = normalized:gsub(source, replacement)
  end
  return normalized:gsub("[^%w]+", "-"):gsub("^%-", ""):gsub("%-$", "")
end

local function supports_report(script)
  return not script:match("dtype_api_smoke%.lua$")
    and not script:match("spill_cleanup_smoke%.lua$")
end

log(string.format("Suite de benchmarks '%s': %d étape(s)", suite, #selected))
for index, entry in ipairs(selected) do
  local command = {
    "cd", shell_quote(root), "&&", shell_quote(mimir_bin),
    "--lua", shell_quote(entry[2]),
  }
  local child_args = {}
  for i = 3, #entry do child_args[#child_args + 1] = entry[i] end
  local detail_report = ""
  if supports_report(entry[2]) then
    detail_report = FS.join(report_dir, report_slug(entry[1]) .. ".json")
    child_args[#child_args + 1] = "--report"
    child_args[#child_args + 1] = detail_report
  end
  if #child_args > 0 then
    command[#command + 1] = "--"
    for _, child_arg in ipairs(child_args) do command[#command + 1] = shell_quote(child_arg) end
  end

  log(string.format("\n[%d/%d] %s", index, #selected, entry[1]))
  local started = Mimir.Benchmark.now()
  local ok, reason, code = os.execute(table.concat(command, " "))
  local passed = ok == true or ok == 0
  summary[#summary + 1] = {
    name=entry[1], passed=passed, seconds=Mimir.Benchmark.now() - started, report=detail_report,
  }
  if not passed then
    failures[#failures + 1] = string.format(
      "%s (%s %s)", entry[1], tostring(reason), tostring(code))
    if not has("--keep-going") then break end
  end
end

local summary_path = FS.join(report_dir, "summary.md")
local summary_file = assert(io.open(summary_path, "w"))
summary_file:write("# Suite de benchmarks `", suite, "`\n\n")
summary_file:write("Généré le ", os.date("!%Y-%m-%dT%H:%M:%SZ"), ".\n\n")
summary_file:write("| Benchmark | Statut | Durée murale | Rapport |\n")
summary_file:write("|---|---:|---:|---|\n")
for _, item in ipairs(summary) do
  local report_name = item.report:match("[^/\\]+$") or item.report
  local link = item.report ~= "" and ("[JSON](" .. report_name .. ")") or "-"
  summary_file:write(string.format("| %s | %s | %.3f s | %s |\n",
    item.name, item.passed and "PASS" or "FAIL", item.seconds, link))
end
summary_file:close()
log("Résumé: " .. summary_path)

if #failures > 0 then
  error("benchmarks en échec: " .. table.concat(failures, ", "))
end

log(string.format("\nSuite '%s' terminée avec succès.", suite))
