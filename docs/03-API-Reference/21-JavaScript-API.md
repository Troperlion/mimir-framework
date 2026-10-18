# Bridge JavaScript

Le bridge utilise Node.js avec l’option `--import` (validé avec Node.js 22.22.1).
Il expose la surface native enregistrée par le bridge de référence :
`Mimir.Model`, `Architectures`, `Layers`, `Tokenizer`, `Dataset`, `Database`,
`Memory`, `MemoryGuard`, `Allocator`, `Htop`, `Viz`, `Serialization`, `Benchmark`
et les autres fonctions de `Mimir`. La découverte est automatique : aucun proxy
factice ni catalogue de fonctions dupliqué. Les fichiers de scripting Lua ne sont
pas modifiés. Le bridge JS réutilise leurs callbacks C++ dans un état Lua privé ;
il dépend donc de Lua, déjà obligatoire dans le framework.

```sh
./bin/mimir --js scripts/examples/bridge_smoke.js
```

Chaque appel est synchrone et exécuté immédiatement dans le processus Mímir.
Les sorties et erreurs proviennent des callbacks natifs ; aucun cache de paramètres
ni traitement différé après la fin du script. Le contexte JS conserve modèle,
tokenizer, dataset et moniteur entre exécutions. Le contexte Lua préexistant est
restauré après l'exécution. Chaque exécution lance une nouvelle VM Node.js : les
variables JS et le cache des modules ne persistent pas entre exécutions.
Les exécutions JS sont sérialisées ; l'utilisation
concurrente des singletons Lua/JS depuis plusieurs threads n'est pas supportée.

## Conventions

- `arg[0]` : chemin du script, puis les arguments utilisateur. En CLI JS,
  `--` placé immédiatement après le chemin est consommé comme séparateur.
- `CONF`, `CONF_PATH`, `CONF_DIR` : configuration injectée par `setSystemConfig`
  ou le mode `--conf` ; valeurs par défaut `{}`, `''`, `''`.
- Aliases historiques : `model`, `architectures`, `tokenizer`, `dataset`, `Memory`,
  `MemoryGuard`, `Allocator`, `htop`, `viz`. Les namespaces sont également exposés
  directement (`Model`, `Tokenizer`, etc.).
- Tableaux JS / tableaux typés : tables natives indexées à partir de 1. Les tableaux
  de sortie JS sont indexés à partir de 0. Les `Buffer` Node sont acceptés comme
  tableaux de bytes. Les `Float32Array` et `Float64Array` préservent leur type
  flottant natif, y compris avec des valeurs entières. Les indices métier transmis explicitement
  (par exemple `Dataset.get(1)`) conservent le contrat Lua.
- Un retour natif : valeur JS ; plusieurs retours : tableau de valeurs ; aucun
  retour : `undefined`. `nil` devient `null`.
- Les erreurs natives levées et les retours `nil, message` / `false, message`
  deviennent des exceptions JS. `fonction.results(...args)` permet de récupérer
  tous les retours bruts, y compris ces tuples d'erreur.
- Les fonctions retournées par les builders (`Database.load(...).cache(...)`)
  restent appelables pendant l'exécution du script.
- Compatibilité JS : `Architectures.create` reste un alias de `Model.create` ;
  `Architectures.dtypes()` conserve les aliases sous forme de tableaux de chaînes.
- `loadScript()` exécute le fichier original, avec son `import.meta.url`, sa
  résolution des imports, son éventuel hashbang et les conventions Node.js
  ESM/CommonJS. Le bridge est préchargé depuis un module temporaire distinct ;
  le répertoire du script peut être en lecture seule.
- `executeScript(code)` utilise un module temporaire à côté de `scriptPath()`
  (ou dans le répertoire courant) pour conserver la résolution relative des imports.
  Pour cette forme inline, le répertoire doit être inscriptible.

```js
const cfg = Mimir.Architectures.default_config('basic_mlp');
Mimir.Model.create('basic_mlp', cfg);
Mimir.Model.allocate_params();
Mimir.Model.init_weights('he', 42);
console.log(Mimir.Model.total_params());

try {
  Mimir.Architectures.default_config('inconnue');
} catch (error) {
  console.error(error.message);
}
```

## Configuration

Le mode `--conf` accepte `js.script` ou `js.scripts`, également dans `run.js` ou
une tâche sélectionnée par `--run`, selon les mêmes conventions de sélection
que les tâches existantes. Les chemins de scripts restent relatifs au répertoire
courant, comme pour Lua.

```json
{"js":{"scripts":[{"script":"scripts/examples/bridge_smoke.js","args":["example"]}]}}
```

## Transport et limites

Node.js communique avec le processus C++ via un canal POSIX dédié, distinct de
stdin/stdout/stderr. Les arguments ne passent ni par un shell ni par des variables
d'environnement globales. L'implémentation actuelle cible les plateformes POSIX.

Le transport représente les valeurs JSON et les fonctions natives retournées.
Les tables natives vides deviennent `{}` (Lua ne distingue pas tableau vide et
objet vide). Les callbacks JS, objets cycliques, `BigInt`, symboles, `DataView`, nombres non finis et
userdata ne constituent pas des arguments supportés. Les entiers au-delà de la
précision de `Number` restent soumis à la limite numérique JavaScript. Les chaînes
transportées doivent être UTF-8. La profondeur des valeurs est limitée à 64.
La parité concerne les API de pilotage natives, pas la bibliothèque standard Lua
ni les modules applicatifs `.lua`. Les appels volumineux ont le coût de copie JSON.

Les tests `BridgeTest.API` vérifient les résultats synchrones, les erreurs, les
aliases, les arguments Unicode, la configuration, le tokenizer, les builders et
la conservation des contextes, le forward/backward, un aller-retour de checkpoint,
les lectures de dataset RGB/texte/cache, les fichiers en lecture seule et CommonJS.
`BridgeTest.JSConf` vérifie les entrées CLI `js`, `run.js`, `--run`, le séparateur
`--`, l’identité du script et les hashbangs ;
`BridgeTest.JSSmoke` exécute l’exemple existant. Ils ne constituent pas une validation graphique
de tous les backends Viz ni de toutes les architectures.
