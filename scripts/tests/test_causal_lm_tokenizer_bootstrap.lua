local Bootstrap = dofile(ROOTWORK.."/scripts/modules/causal_lm_tokenizer.lua")
local path = "/tmp/mimir_causal_lm_tokenizer_bootstrap.json"
os.remove(path)

local corpus = table.concat({
  "alpha beta gamma delta epsilon",
  "the quick brown fox jumps over the lazy dog",
  "un petit corpus causal reproductible",
}, "\n")

-- Missing tokenizer: generate, validate and persist one.
local tokens, status = Bootstrap.ensure({
  corpus = corpus,
  path = path,
  vocab_size = 64,
  padding_idx = 0,
  seq_len = 8,
})
assert(type(tokens) == "table" and #tokens > 8)
assert(status.generated == true)
assert(status.replaced_loaded_tokenizer == false)
local saved = io.open(path, "rb")
assert(saved, "generated tokenizer was not persisted")
saved:close()

-- Existing but incompatible tokenizer capacity: replace it with a model-specific one.
assert(Mimir.Tokenizer.create(128))
assert(Mimir.Tokenizer.ensure_vocab_from_text(corpus))
assert(Mimir.Tokenizer.save(path))

local replaced_tokens, replaced = Bootstrap.ensure({
  corpus = corpus,
  path = path,
  vocab_size = 64,
  padding_idx = 0,
  seq_len = 8,
})
assert(type(replaced_tokens) == "table" and #replaced_tokens > 8)
assert(replaced.generated == true)
assert(replaced.replaced_loaded_tokenizer == true)
assert(tonumber(Mimir.Tokenizer.get_max_vocab()) == 64)
assert(tonumber(Mimir.Tokenizer.vocab_size()) <= 64)

os.remove(path)
print("PASS causal LM tokenizer bootstrap")
