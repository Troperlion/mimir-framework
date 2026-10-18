-- Run with: ./bin/mimir --lua scripts/tests/test_tools_help.lua
-- Help must finish before opening data, starting commands or requiring Mimir.
local scripts = {
  "analyze_model", "build_tags_vocab", "convert_checkpoint2safetensor",
  "convert_safetensors2raw_folder", "inspect_architectures",
  "inspect_z_prior_raw_folder", "show-graph", "build_mpk", "export_arch_mpk",
  "compile_mpk", "load_mpk", "mpk_node_wizard", "add_vision_mpk_architectures",
}
local original = {arg=arg, print=print, exit=os.exit, open=io.open,
  execute=os.execute, popen=io.popen, mimir=Mimir, stdout=io.stdout}
local exited = {}
local function forbidden() error("help attempted a data or runtime operation") end
local ok, err = pcall(function()
  _G.Mimir = nil
  io.open, io.popen, os.execute = forbidden, forbidden, forbidden
  os.exit = function(code) assert(code == 0); error(exited) end
  for _, name in ipairs(scripts) do
    for _, flag in ipairs({"--help", "-h"}) do
      local lines = {}
      print = function(line) lines[#lines + 1] = tostring(line) end
      io.stdout = {write = function(_, line) lines[#lines + 1] = line end}
      arg = {flag}
      local done, reason = pcall(dofile, ROOTWORK.."/scripts/tools/"..name..".lua")
      assert(not done and reason == exited, name..": "..tostring(reason))
      local output = table.concat(lines, "\n")
      assert(output:find("Options:", 1, true), name)
      assert(output:find("Exemples:", 1, true), name)
      assert(not output:find("Options detectees:", 1, true), name)
    end
  end
end)
arg, print, os.exit, io.open = original.arg, original.print, original.exit, original.open
os.execute, io.popen, _G.Mimir = original.execute, original.popen, original.mimir
io.stdout = original.stdout
assert(ok, err)
print("test_tools_help: OK (26 checks, no data access or runtime action)")
