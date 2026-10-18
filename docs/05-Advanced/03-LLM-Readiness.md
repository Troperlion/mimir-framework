# État de préparation aux LLM

Optimiser, diagnostiquer et stabiliser des runs complexes.

**Public concerné :** Utilisateur intermédiaire à avancé.

> **Prérequis**
>
> Avoir déjà exécuté au moins un pipeline complet.

Cette page décrit ce qui est **effectivement** en place dans le codebase pour un LLM, et ce qui manque pour une génération utile au quotidien.

Point d’entrée conseillé : [04-Source-Code-Map.md](04-Source-Code-Map.md).

## 1) Ce qui existe déjà

- Une architecture decoder-only native : `src/Models/NLP/CausalLMModel.cpp`,
  avec RMSNorm, RoPE, GQA, SwiGLU et poids embedding/LM head partagés.
- Une architecture encodeur Transformer : `src/Models/NLP/TransformerModel.cpp`.
- Un chemin tokens entiers via `__input__` pour `causal_lm`.
- Des kernels CPU pour attention et matmul (OpenMP/SIMD selon build) : `src/runtimes/cpu/LayerOps.hpp`, `src/runtimes/cpu/SIMD_Ops.hpp`.
- Une API tokenizer en Lua via `Mimir.Tokenizer.*` et un bootstrap de
  tokenizer compatible dans `scripts/modules/causal_lm_tokenizer.lua`.

## 2) Ce que `Mimir.Model.infer()` fait vraiment (important)

`Mimir.Model.infer(prompt)` est présent, mais aujourd’hui ce n’est pas une “generation” autoregressive complète :

- côté C++ : `src/scriptings/Lua/luaScripting/LuaScripting.cpp` tokenize une string si un tokenizer existe, encode si un encoder existe, puis appelle `Model::forward(output)`.
- la “décode” ensuite via `Model::eval(...)`.

Or `Model::eval(...)` (voir `src/Model.cpp`) produit actuellement :

- un MSE trivial sur un `target` binaire,
- puis des logits uniformes + des tokens top-k “factices” (0..7) si un tokenizer est présent.

Conclusion : **ne pas** considérer `infer()` comme une génération LLM valide pour l’instant.

## 3) Limites actuelles pour un LLM “pratique”

- Pas de frontend Lua de génération autoregressive/sampling pour `causal_lm`.
- Pas d’API runtime de sampling (top-k/top-p/temperature) côté C++ ; si vous voulez sampler, ça doit être implémenté côté scripts Lua pour l’instant.
- Offload GPU : l’offload Vulkan/OpenCL est ciblé sur `Linear` en inférence, pas une stack LLM complète (voir `src/Model.cpp`).

## 4) Roadmap minimale (concrète)

Pour passer de “Transformer entraînable” à “LLM utilisable” :

1) Exposer des logits utilisables (dernier layer / projection vocab) via une API stable.
2) Ajouter une API `prefill` + `decode` avec KV-cache (structures + sérialisation).
3) Ajouter un module de sampling (au moins greedy + top-k) et une boucle de génération.
4) Optimiser attention/matmul et/ou étendre l’offload.

## 5) Où regarder dans le code

- Registry + config merge : `src/Models/Registry/ModelArchitectures.*`.
- Builder Transformer : `src/Models/NLP/TransformerModel.cpp`.
- Ops d’attention : `src/runtimes/cpu/LayerOps.hpp`.
- Bindings Lua (forward, infer, encode prompt) : `src/scriptings/Lua/luaScripting/LuaScripting.cpp`.

Et côté scripts (utile pour comprendre l’intention et l’usage actuel) :

- `scripts/examples/example_conf_inference.lua` : inférence config-driven sur architecture transformer.
- `scripts/benchmarks/benchmark_attention.lua` : cas transformer/attention en benchmark (dont mode causal).
- `scripts/templates/template_pipeline_only.lua` et `scripts/templates/template_pipeline_args.lua` : pipelines template pour workflows autoregressifs.

## Étapes suivantes

- [Page précédente : Debugging & stabilité numérique](02-Debugging.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Advanced — Carte du code source (C/C++)](04-Source-Code-Map.md)
