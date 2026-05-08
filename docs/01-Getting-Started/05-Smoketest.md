
# Smoketest (5 minutes)

Objectif : vérifier rapidement que le build, la CLI et les chemins essentiels (Lua, forward, sérialisation) fonctionnent sur ta machine.

## 1) Build propre

```bash
cmake -S . -B build
cmake --build build -j
```

## 2) Vérifier la CLI

```bash
./bin/mimir --help
```

## 2bis) Tester le mode `--conf` (JSON -> scripts)

Si `config.json` contient une section `lua.scripts`, tu peux valider l’injection de conf + exécution :

```bash
./bin/mimir --conf config.json
```

## 3) Exécuter un exemple minimal (Lua)

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

## 4) Tester une surface API un peu plus large

### Ops / layers

```bash
./bin/mimir --lua scripts/tests/test_list_archi_conf.lua
```

### Sérialisation

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

## 5) Lancer un petit test suite Lua (recommandé)

Tests rapides :

```bash
./bin/mimir --lua scripts/tests/test_list_archi_conf.lua
./bin/mimir --lua scripts/tests/test_vae_conv_generate.lua
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

## 6) (Optionnel) Exécuter les tests CMake/CTest si présents

Si ton build a généré des targets de test :

```bash
cd build
ctest --output-on-failure
```

## Notes

- Si tu vois des refus MemoryGuard/OOM, lis `docs/02-User-Guide/09-Memory.md`.
- Si tu veux faire un run plus lourd : `./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1`.