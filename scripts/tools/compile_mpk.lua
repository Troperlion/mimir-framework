---@diagnostic disable: undefined-global

-- Compile a modern MPK pseudocode source to opaque typed binary-v4.

dofile(ROOTWORK.."/scripts/modules/mpk_help.lua").show("compile_mpk")

local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")
local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")

local function die(msg)
  io.stderr:write("[compile_mpk] " .. tostring(msg) .. "\n")
  os.exit(1)
end


local opts = Args.parse(arg) or {}

local input = Args.get_str(opts, "in", "")
local output = Args.get_str(opts, "out", "")
if input == "" then die("missing --in <source.mpk>") end
if output == "" then die("missing --out <compiled.mpk.bin>") end

local parent = FS.dirname(output)
if parent and parent ~= "" then FS.mkdir_p(parent) end

local ok, err = MPK.compile(input, output)
if not ok then die(err) end

local pkg, read_err = MPK.read(output)
if not pkg then die("compiled verification failed: " .. tostring(read_err)) end

io.stdout:write("[compile_mpk] OK\n")
io.stdout:write("  source:    " .. input .. "\n")
io.stdout:write("  output:    " .. output .. "\n")
io.stdout:write("  container: " .. tostring(pkg.container) .. "\n")
io.stdout:write("  version:   binary-v4\n")
io.stdout:write("  name:      " .. tostring(pkg.header and pkg.header.name) .. "\n")
