#!/usr/bin/env mimir --lua
-- Vérifie création de spill puis nettoyage APRÈS sortie du processus enfant.
local Help = dofile(ROOTWORK..'/scripts/modules/help_cli.lua')
Help.auto_exit_help()
local FS = dofile(ROOTWORK..'/scripts/modules/fs.lua')
local B = dofile(ROOTWORK..'/scripts/benchmarks/common.lua')
if B.has('--worker') then
  -- Chaque matrice tient en mémoire; l'ensemble dépasse le budget de 4 MiB.
  assert(Mimir.Allocator.configure({max_ram_gb=4/1024,enable_compression=false}))
  B.create('basic_mlp', {input_dim=512,hidden_dim=512,output_dim=512,hidden_layers=8})
  local count = #FS.list_dir('.mimir-spill')
  assert(count > 0, 'aucun spill observé: test non exercé')
  local marker = assert(io.open('spill-observed', 'w'))
  marker:write(tostring(count)); marker:close()
  log('Spill observé: '..count..' fichier(s)')
else
  assert(not FS.is_windows(), 'Ce lanceur de processus nécessite un shell POSIX')
  local work = os.tmpname()
  assert(os.remove(work))
  assert(FS.mkdir_p(work))
  local bin = os.getenv('MIMIR_BIN') or (ROOTWORK..'/bin/mimir')
  if bin:sub(1,1) ~= '/' then bin=ROOTWORK..'/'..bin end
  local command = 'cd '..FS.quote(work)..' && '..FS.quote(bin)..' --lua '..
    FS.quote(ROOTWORK..'/scripts/benchmarks/spill_cleanup_smoke.lua')..' -- --worker'
  local ok, reason, code = os.execute(command)
  local passed = ok == true or ok == 0
  local observed = FS.file_exists(work..'/spill-observed')
  local remaining = #FS.list_dir(work..'/.mimir-spill')
  -- Répertoire temporaire possédé par ce test, conservé en cas d'échec.
  if passed and observed and remaining == 0 then
    assert(os.remove(work..'/spill-observed'))
    if FS.is_dir(work..'/.mimir-spill') then assert(os.remove(work..'/.mimir-spill')) end
    if FS.is_dir(work..'/logs') then
      for _, name in ipairs(FS.list_dir(work..'/logs')) do
        assert(os.remove(work..'/logs/'..name))
      end
      assert(os.remove(work..'/logs'))
    end
    assert(os.remove(work))
  else
    error(string.format('spill cleanup: child=%s/%s/%s observed=%s remaining=%d artifacts=%s',
      tostring(ok),tostring(reason),tostring(code),tostring(observed),remaining,work))
  end
  log('PASS spill créé dans l’enfant et nettoyé après sa sortie')
end
