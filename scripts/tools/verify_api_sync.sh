#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 - "$project_root" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
source = (root / "src/scriptings/Lua/luaScripting/LuaScripting.cpp").read_text(encoding="utf-8")
stub = (root / "mimir-api.lua").read_text(encoding="utf-8")

# registerAPI() construit les tables publiques. Une fonction C suivie d'un
# lua_setfield expose le nom indiqué; les autres lua_setfield remplissent des
# objets de données et ne font pas partie de ce contrôle.
pairs = re.findall(
    r'lua_pushcfunction\(L,\s*([A-Za-z0-9_]+)\);\s*'
    r'lua_setfield\(L,\s*-[0-9]+,\s*"([A-Za-z0-9_]+)"\);',
    source,
)
registered = sorted({field for _, field in pairs})
documented = {
    qualified.rsplit(".", 1)[-1]
    for qualified in re.findall(
        r'^function\s+(Mimir(?:\.[A-Za-z0-9_]+)+)\s*\(', stub, re.MULTILINE
    )
}
missing = [name for name in registered if name not in documented]

required_modules = {
    "Model", "Architectures", "Layers", "Tokenizer", "Dataset", "Database",
    "IO", "Memory", "MemoryGuard", "Allocator", "Htop", "Viz",
    "Serialization",
}
top_level = set(re.findall(r'^---@field\s+([A-Za-z0-9_]+)\s+Mimir[A-Za-z0-9_]+API\b', stub, re.MULTILINE))
missing_modules = sorted(required_modules - top_level)

if missing or missing_modules:
    if missing:
        print("API C++ absente du stub EmmyLua: " + ", ".join(missing), file=sys.stderr)
    if missing_modules:
        print("Modules EmmyLua manquants: " + ", ".join(missing_modules), file=sys.stderr)
    raise SystemExit(1)

print(f"API EmmyLua synchronisée: {len(registered)} noms de fonctions enregistrés, "
      f"{len(required_modules)} modules publics vérifiés")
PY
