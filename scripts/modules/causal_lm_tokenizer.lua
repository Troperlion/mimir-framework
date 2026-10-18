---@class MimirCausalTokenizerModule
local M = {}

local function readable(path)
  if type(path) ~= "string" or path == "" then return false end
  local file = io.open(path, "rb")
  if not file then return false end
  file:close()
  return true
end

local function tokenizer_compatibility(model_vocab, padding_idx, probe_text)
  local actual = tonumber(Mimir.Tokenizer.vocab_size())
  local maximum = tonumber(Mimir.Tokenizer.get_max_vocab())
  if not actual or actual < 7 then
    return false, "vocabulary is missing required special tokens"
  end
  if actual > model_vocab then
    return false, string.format("vocabulary has %d entries, model supports %d", actual, model_vocab)
  end
  if maximum and maximum > model_vocab then
    return false, string.format("tokenizer capacity is %d, model supports %d", maximum, model_vocab)
  end

  local specials = {
    pad = tonumber(Mimir.Tokenizer.pad_id()),
    unk = tonumber(Mimir.Tokenizer.unk_id()),
    seq = tonumber(Mimir.Tokenizer.seq_id()),
    mod = tonumber(Mimir.Tokenizer.mod_id()),
    mag = tonumber(Mimir.Tokenizer.mag_id()),
  }
  for name, id in pairs(specials) do
    if not id or id < 0 or id >= model_vocab then
      return false, string.format("invalid %s token id: %s", name, tostring(id))
    end
  end
  if specials.pad ~= padding_idx then
    return false, string.format(
      "PAD id %d differs from model padding_idx %d", specials.pad, padding_idx)
  end

  local tokens, token_err = Mimir.Tokenizer.tokenize(probe_text)
  if type(tokens) ~= "table" then
    return false, "probe tokenization failed: " .. tostring(token_err)
  end
  for i, id in ipairs(tokens) do
    id = tonumber(id)
    if not id or id < 0 or id >= model_vocab or id % 1 ~= 0 then
      return false, string.format("probe token #%d is outside model vocabulary: %s", i, tostring(id))
    end
  end
  return true, nil, tokens
end

---@param options MimirCausalTokenizerOptions
---@return TokenIds tokens
---@return MimirCausalTokenizerInfo info
function M.ensure(options)
  assert(type(options) == "table", "causal tokenizer options must be a table")
  local corpus = assert(options.corpus, "causal tokenizer corpus is required")
  local model_vocab = assert(tonumber(options.vocab_size), "causal tokenizer vocab_size is required")
  local padding_idx = tonumber(options.padding_idx) or 0
  local seq_len = tonumber(options.seq_len) or 512
  local path = options.path
  local loaded = false
  local reason

  if readable(path) then
    local ok, err = Mimir.Tokenizer.load(path)
    if ok ~= false then
      loaded = true
      local compatible, why, tokens =
          tokenizer_compatibility(model_vocab, padding_idx, corpus)
      if compatible then
        Mimir.Tokenizer.set_max_length(seq_len)
        return tokens, {
          generated = false,
          path = path,
          vocab_size = tonumber(Mimir.Tokenizer.vocab_size()),
          max_vocab = tonumber(Mimir.Tokenizer.get_max_vocab()),
        }
      end
      reason = why
    else
      reason = "load failed: " .. tostring(err)
    end
  else
    reason = "file is absent or unreadable"
  end

  local created, create_err = Mimir.Tokenizer.create(model_vocab)
  assert(created ~= false, "Tokenizer.create failed: " .. tostring(create_err))
  Mimir.Tokenizer.set_max_length(seq_len)
  local ensured, ensure_err = Mimir.Tokenizer.ensure_vocab_from_text(corpus)
  assert(ensured ~= false, "tokenizer vocabulary creation failed: " .. tostring(ensure_err))

  local merges = math.floor(tonumber(options.bpe_merges) or 0)
  if merges > 0 then
    local learned, learn_err = Mimir.Tokenizer.learn_bpe({corpus}, merges)
    assert(learned ~= false, "tokenizer BPE learning failed: " .. tostring(learn_err))
  end

  local compatible, why, tokens =
      tokenizer_compatibility(model_vocab, padding_idx, corpus)
  assert(compatible, "generated tokenizer is incompatible with model: " .. tostring(why))

  if type(path) == "string" and path ~= "" then
    local saved, save_err = Mimir.Tokenizer.save(path)
    assert(saved ~= false, "generated tokenizer save failed: " .. tostring(save_err))
  end

  return tokens, {
    generated = true,
    replaced_loaded_tokenizer = loaded,
    reason = reason,
    path = path,
    vocab_size = tonumber(Mimir.Tokenizer.vocab_size()),
    max_vocab = tonumber(Mimir.Tokenizer.get_max_vocab()),
  }
end

return M
