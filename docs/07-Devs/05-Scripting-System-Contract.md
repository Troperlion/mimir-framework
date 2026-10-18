# Contrat de scripting entre langages

Implémenter ou modifier des briques techniques sans casser le contrat global.

**Public concerné :** Développeur framework (C/C++/runtime/scripting).

> **Prérequis**
>
> Comprendre le registre d'architectures et les conventions I/O.


Ce chapitre formalise le contrat API systeme que tous les bridges de scripting doivent respecter.

## Etat des bindings

- Lua est le bridge de reference et reste le seul contrat complet et stable aujourd'hui.
- JS expose désormais la surface des callbacks natifs de référence par des appels synchrones Node.js vers C++ ; voir [le contrat JS et ses limites](../03-API-Reference/21-JavaScript-API.md).
- Les bindings C# et Rust restent incomplets.
- La parite progresse par morceaux: certaines APIs sont disponibles, d'autres renvoient encore une erreur explicite ou une sortie partielle.
- Toute documentation ou demo pour un bridge non-Lua doit le signaler clairement.

## 1. Pourquoi un contrat commun

Sans contrat central, chaque langage derive avec ses propres noms globaux et comportements.

Resultat :

- scripts non portables,
- bugs subtils en migration Lua -> Python/Ruby/JS,
- maintenance plus couteuse.

Le contrat commun est defini dans `src/scriptings/ScriptingContext.hpp`.

## 2. Noms systeme canoniques

Exemples normalises :

- namespace: `Mimir`
- conf: `CONF`, `CONF_PATH`, `CONF_DIR`
- args script: `arg`
- aliases: `model`, `architectures`, `tokenizer`, `dataset`, `Memory`, `MemoryGuard`, `Allocator`, `htop`, `viz`

Ces noms ne doivent pas varier entre bridges.

## 3. Interface runtime bridge

La base polymorphe est `src/scriptings/ScriptingRuntime.hpp`.

Contrat minimal :

- `setArgs(...)`
- `loadScript(...)`
- `executeScript(...)`
- `registerAPI()`

Tout nouveau bridge (Python/Ruby/JS/...) doit implementer ce contrat.

## 4. Contexte partage

Le contexte d'execution doit reutiliser `ScriptingContext` pour :

- stocker l'etat runtime (modele courant, tokenizer, encoder),
- garantir les memes conventions de reset/nettoyage,
- fournir un logging coherent.

## 5. Integration mode conf

Quel que soit le langage, l'execution config-driven doit injecter :

- `CONF` (objet config complet),
- `CONF_PATH`,
- `CONF_DIR`,
- `arg`.

Cela assure la compatibilite des workflows.

## 6. Checklist pour un nouveau bridge

1. Heriter de `ScriptingRuntime`.
2. Creer un contexte derive de `ScriptingContext`.
3. Exposer les memes noms systeme (`kGlobal*`, `kAlias*`).
4. Garantir le meme contrat d'erreur.
5. Ajouter un smoke test equivalent au bridge Lua.
6. Tant que le bridge n'a pas la parite fonctionnelle, le documenter comme experimental / incomplet.

## 7. Demo metier - script portable entre bridges

Script type attendu (conceptuellement identique entre Lua/Python/etc.) :

```lua
local conf = CONF or {}
local arch = conf.arch or "vae_conv"
local cfg, err = Mimir.Architectures.default_config(arch)
if not cfg then error(err) end

print("CONF_PATH=" .. tostring(CONF_PATH))
print("CONF_DIR=" .. tostring(CONF_DIR))
```

Ce qui compte metier :

1. Le script lit la conf via les memes noms globaux.
2. Le meme script logique peut etre porte dans un autre bridge.
3. Les erreurs ont une forme exploitable en CI.

## 8. Demo metier - test de non-regression du contrat

Pour chaque bridge, verifier ces points :

1. `Mimir` est visible.
2. `arg` est injecte.
3. `CONF`, `CONF_PATH`, `CONF_DIR` sont injectes en mode `--conf`.
4. aliases (`model`, `tokenizer`, `dataset`, `MemoryGuard`) sont presents.

Exemple de commande (bridge Lua actuel) :

```bash
./bin/mimir --conf config.json
```

Si un bridge ne passe pas cette matrice, il n'est pas pret pour un workflow metier stable.

La matrice de contrat JS est couverte par `BridgeTest.API`, avec configuration également injectable par `--conf`. Les autres bridges restent à valider au cas par cas. Les limites de transport JS sont documentées dans sa référence API.

## 9. Regle d'evolution API

Pour toute evolution du contrat :

1. Ajouter avant de supprimer (deprecation progressive).
2. Documenter la transition dans la doc dev.
3. Fournir un test de compat backward.
4. Eviter tout renommage silencieux des globals/aliases.

## 10. Regle filesystem cross-plateforme

Les scripts metier doivent utiliser `scripts/modules/fs.lua` pour les acces filesystem:

- `FS.mkdir_p`, `FS.join`, `FS.dirname`, `FS.list_dir`, `FS.file_exists`, `FS.is_dir`.

Ne pas introduire de shell Linux-specifique pour le filesystem (`mkdir -p`, `ls`, `find`, `test -d`, `stat`) dans les scripts applicatifs.

Exception: les appels shell qui pilotent des processus externes (ex: ouvrir un navigateur, lancer un binaire tiers) peuvent rester en `os.execute`/`io.popen` si necessaire.

## Étapes suivantes

- [Page précédente : Runtime - Modifier Et Ajouter Un Backend](04-Runtime-Development.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Visualizer: Tips Et Nouvelles Features](06-Visualizer-Tips-And-Features.md)
