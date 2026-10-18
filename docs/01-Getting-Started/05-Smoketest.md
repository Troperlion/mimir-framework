# Valider l’installation

Vérifier en 5 minutes que l'environnement est sain.

**Public concerné :** Débutant et contributeur.

> **Prérequis**
>
> Build effectué au moins une fois.

Ce smoketest vérifie que le build, la CLI et les chemins essentiels — Lua,
passe avant et sérialisation — fonctionnent sur votre machine.

## 1) Build propre

```bash
cmake -S . -B build
cmake --build build -j
```

## 2) Vérifier la CLI

```bash
./bin/mimir --help
```

## 3) Exécuter un exemple minimal (Lua)

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

## 4) Tester une surface API un peu plus large

### Ops / layers

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
```

### Sérialisation

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

## 5) Lancer un petit test suite Lua (recommandé)

Tests rapides :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
./bin/mimir --lua scripts/tests/test_vae_conv_generate.lua
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

## 6) (Optionnel) Exécuter les tests CMake/CTest si présents

Si votre build a généré des targets de test :

```bash
cd build
ctest --output-on-failure
```

## Notes

- Si vous voyez des refus MemoryGuard/OOM, lis `docs/02-User-Guide/09-Memory.md`.
- Si vous voulez faire un run plus lourd : `./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1`.

## Étapes suivantes

- [Page précédente : Organisation du dépôt](04-Repo-Layout.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Parcours par niveau de compétence](06-Learning-Paths.md)
