# CLI et points d’entrée

Cette page décrit le parsing de la ligne de commande, les chemins d’exécution
du binaire `mimir` et les conventions transmises aux systèmes de scripting.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

Source de vérité :

- Entry point : `src/main.cpp`
- Scripting Lua : `src/scriptings/Lua/luaScripting/LuaScripting.hpp` et `src/scriptings/Lua/luaScripting/LuaScripting.cpp`
- Contrat API système partagé : `src/scriptings/ScriptingContext.hpp`
- Config → modèle : `src/Models/Registry/ModelArchitectures.hpp` et `src/Models/Registry/ModelArchitectures.cpp`
- Sécurité mémoire : `src/MemorySafety.hpp` + `src/MemoryGuard.hpp`

## 1) Options supportées

Dans `src/main.cpp`, la CLI expose :

- `--lua <script.lua>` : exécute un script Lua.
- `--config <config.json>` : crée un modèle depuis un JSON (via le registre) puis init (chemin starter/legacy).
- `--conf <config.json>` : charge la config et exécute les scripts `lua.scripts` (chemin recommandé pour workflows).
- `--help` : usage.

Les arguments après `--lua script.lua` sont passés au script (injectés comme `arg`).

## 2) Séquence de boot

Au démarrage :

- bannière de version,
- `MemorySafety::validateLegacyDisabled()` + `runMemoryIntegrityTest()`,
- affichage capacités CPU (AVX2/FMA/F16C/BMI2),
- puis dispatch selon les arguments.

## 3) Path `--lua`

- vérifie l’existence du fichier,
- instancie `LuaScripting`, injecte args, charge et exécute.

## 4) Path `--config`

- lit JSON,
- choisit `architecture` (default `t2i_autoencoder`),
- part de `ModelArchitectures::defaultConfig(name)` puis applique les overrides:
  - section `model` si présente,
  - section `config[arch_type]` si présente,
- `ModelArchitectures::create`, puis `allocateParams` + `initializeWeights("he")`.

## 5) Path `--conf`

- lit JSON,
- résout `lua.scripts` (ou `run.lua.scripts`),
- injecte `CONF`, `CONF_PATH`, `CONF_DIR`,
- exécute chaque script Lua dans l'ordre, avec ses `args` éventuels.

## Étapes suivantes

- [Page précédente : Internals : registre d’architectures (`ModelArchitectures`) et builders](19-Models-Registry-And-Builders.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : GPU Runtimes — CUDA & ROCm (C++)](21-GPU-Runtimes.md)
