# Ajouter une entrée de scripting Python

Contribuer avec des changements cohérents et maintenables.

**Public concerné :** Contributeur du projet.

> **Prérequis**
>
> Connaître le workflow Git et les bases du projet.


Ce guide montre comment ajouter une entrée Python minimale en reprenant le design de l'intégration Lua.

But :

- exécuter `./bin/mimir --python script.py -- args...`,
- exposer une API framework proche de `Mimir.*`,
- conserver la compatibilité avec le mode config-driven (`--conf`).

## Sur cette page

- [1. Préparation](#1-préparation)
- [2. Créer le bridge Python](#2-créer-le-bridge-python)
- [3. Exposer l'API framework](#3-exposer-lapi-framework)
- [4. Ajouter l'entrée CLI](#4-ajouter-lentrée-cli)
- [5. Compatibilité mode conf](#5-compatibilité-mode-conf)
- [6. Smoke test minimal](#6-smoke-test-minimal)
- [7. Plan d'implémentation incrémental](#7-plan-dimplémentation-incrémental)
- [8. Pseudo-code de robustesse (erreurs)](#8-pseudo-code-de-robustesse-erreurs)
- [9. Étendre vers d'autres langages](#9-étendre-vers-dautres-langages)
- [10. Checklist de revue PR](#10-checklist-de-revue-pr)
- [Étapes suivantes](#étapes-suivantes)

## 1. Préparation

Référence existante (source de vérité) :

- `src/scriptings/ScriptingRuntime.hpp`
- `src/scriptings/ScriptingContext.hpp`
- `src/scriptings/Lua/luaScripting/LuaScripting.hpp`
- `src/scriptings/Lua/luaScripting/LuaScripting.cpp`
- `src/main.cpp`

Approche recommandée : démarrer simple, puis étendre.

MVP (minimum viable product) :

1. création d'un bridge C++ pour Python,
2. exécution d'un script,
3. injection des arguments script,
4. exposition de 2-3 APIs critiques (`Model.create`, `Model.forward`, `Model.save`),
5. propagation d'erreur avec codes de retour cohérents.

## 2. Créer le bridge Python

Créer deux nouveaux fichiers :

- `src/PythonScripting.hpp`
- `src/PythonScripting.cpp`

### 2.1 Squelette d'interface

Inspirer l'API publique de `LuaScripting` :

```cpp
class PythonScripting {
public:
    PythonScripting();
    ~PythonScripting();

    void setArgs(const std::string& script_path,
                 const std::vector<std::string>& script_args);

    bool loadScript(const std::string& filepath);
    bool executeScript(const std::string& code);

    void registerAPI();
};
```

### 2.2 Points d'implémentation

- Initialiser le runtime Python au constructeur.
- Fermer proprement au destructeur.
- Capturer les exceptions Python et les convertir en erreurs C++ lisibles.
- Implémenter un objet contexte global hérité de `ScriptingContext` (modèle courant, tokenizer courant, config courante).

## 3. Exposer l'API framework

Objectif : garder une interface proche de Lua pour limiter les différences entre langages.

### 3.1 Surface minimale conseillée

- `Mimir.Model.create(...)`
- `Mimir.Model.allocate_params()`
- `Mimir.Model.init_weights()`
- `Mimir.Model.forward(...)`
- `Mimir.Model.save(...)`
- `Mimir.Architectures.available()`

### 3.2 Mapping recommandé

- Garder les mêmes noms que dans Lua quand c'est possible.
- Conserver les mêmes conventions de types (tableau float, tableau int, bool, string).
- Conserver les mêmes messages d'erreur et préconditions.

## 4. Ajouter l'entrée CLI

Fichier cible :

- `src/main.cpp`

Ajouter une branche de parsing similaire à `--lua` :

```cpp
if (mode == "--python") {
    PythonScripting py;
    py.setArgs(scriptPath, scriptArgs);
    ok = py.loadScript(scriptPath);
}
```

Contraintes UX importantes :

- conserver le séparateur `--` pour les arguments du script,
- conserver le même contrat de retour (0 succès, non-0 échec),
- logguer une erreur explicite si le script ne peut pas être chargé.

## 5. Compatibilité mode conf

Si l'exécution est lancée avec `--conf ...`, injecter aussi côté Python les variables standard :

- `CONF`
- `CONF_PATH`
- `CONF_DIR`
- `arg`

Attendu : les scripts Python puissent fonctionner dans les mêmes workflows que les scripts Lua.

Recommandation : réutiliser les noms canoniques définis dans `ScriptingContext` (`kGlobalNamespace`, `kGlobalConf`, `kGlobalConfPath`, `kGlobalConfDir`, `kGlobalArg`, `kAlias*`).

## 6. Smoke test minimal

Créer un script de test, exemple :

- `scripts/tests/test_python_entry_smoke.py`

Exemple de scénario :

1. créer un modèle simple via registre,
2. allouer + init,
3. faire un forward sur un mini tenseur,
4. sauvegarder un checkpoint dans `checkpoint/`.

Commande attendue :

```bash
./bin/mimir --python scripts/tests/test_python_entry_smoke.py -- --dry-run
```

## 7. Plan d'implémentation incrémental

### Étape A (MVP)

- bridge Python + CLI + exécution script,
- API minimaliste model,
- smoke test.

### Étape B (Parité fonctionnelle)

- ajouter Architectures, Dataset, Tokenizer,
- ajouter helpers mémoire/monitoring,
- aligner les erreurs et conventions Lua.

### Étape C (industrialisation)

- tests non-régression multi-scripts,
- doc API Python dédiée,
- benchmark de surcharge du bridge.

## 8. Pseudo-code de robustesse (erreurs)

```cpp
bool PythonScripting::loadScript(const std::string& path) {
    try {
        // 1) charger le fichier
        // 2) exécuter
        // 3) vérifier la fin sans exception
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erreur Python: " << e.what() << std::endl;
        return false;
    }
}
```

## 9. Étendre vers d'autres langages

Le même pattern s'applique à Ruby, JavaScript, Perl, Java, Rust, etc. :

1. créer un bridge dédié (ex: `RubyScripting`, `JsScripting`),
2. reproduire le contrat de `ScriptingRuntime` + `ScriptingContext` (args, API système, erreurs),
3. brancher une entrée CLI dédiée,
4. valider avec un smoke test équivalent.

Principe central : un seul noyau framework, plusieurs portes d'entrée scripting avec un contrat homogène.

## 10. Checklist de revue PR

- API bridge stable et documentée,
- erreurs converties proprement,
- `--conf` compatible,
- smoke test reproductible,
- doc Contributing et index mis à jour.

## Étapes suivantes

- [Page précédente : Chapitre développeur complet : Étendre le framework](03-Extending-Models-Runtimes-And-Features.md)
- [Index de la documentation](../00-INDEX.md)
- [Revenir à la documentation](../00-INDEX.md)
