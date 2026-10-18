# Modifier ou ajouter un runtime

Ce tutoriel suit le routage actuel des layers :
`Model` appelle `RuntimeRouter`, qui planifie et essaie les runtimes dans
l'ordre ROCm, CUDA, Vulkan, OpenCL, CPU.

## Sources de vérité

- `src/runtimes/AbstractRuntime.hpp`
- `src/runtimes/AbstractRuntime.cpp`
- `src/runtimes/RuntimeRouter.hpp`
- `src/runtimes/RuntimeRouter.cpp`
- `src/runtimes/cpu/CpuRuntime.cpp`
- `src/runtimes/cpu/RuntimeLayerDispatch.hpp`
- `src/Model.cpp`
- `CMakeLists.txt`

## Comprendre le contrat

Un runtime dérive de `AbstractRuntime` et fournit au minimum :

```cpp
const char* name() const override;
bool initialize(const RuntimeConfig& cfg) override;
void shutdown() override;
bool isInitialized() const override;
bool linearForward(/* ... */) override;
bool forwardLayer(/* ... */) override;
```

La passe arrière générique possède une implémentation de base, mais un backend
peut redéfinir `backwardLayer`. Les votes
`supportsForwardLayerType` et `supportsBackwardLayerType` servent à composer
les routes avant l'exécution.

`forwardLayer` retourne :

- `true` uniquement si toutes les sorties demandées sont produites ;
- `false` si le runtime est inactif, refuse les formes reçues ou ne possède pas
  d'implémentation applicable.

`false` est donc un refus contrôlé. Il permet au routeur d'essayer le runtime
suivant.

## Modifier un backend existant

Prenez le `case` équivalent dans le runtime CPU comme référence mathématique.
Pour une modification :

1. vérifiez le vote de support ;
2. vérifiez les flags de `RuntimeConfig` ;
3. validez le nombre d'entrées, les pointeurs et les formes ;
4. ne modifiez pas `outputs` avant de savoir que le cas est accepté, ou
   reconstruisez entièrement la sortie avant de retourner `true` ;
5. comparez le résultat au CPU avec une tolérance explicite ;
6. vérifiez séparément l'entraînement et l'inférence.

> **Attention**
> Le fait qu'un runtime vote pour un `LayerType` ne prouve pas qu'un kernel GPU
> spécialisé sera utilisé pour toutes les formes. CUDA et ROCm possèdent aussi
> des chemins partagés de dispatch.

## Ajouter un backend

Un nouveau backend exige plus qu'une classe :

1. créez `src/runtimes/<backend>/MyRuntime.hpp` et `.cpp` ;
2. implémentez l'initialisation, l'arrêt, les votes et le dispatch ;
3. ajoutez une option CMake et la détection de ses dépendances ;
4. ajoutez ses sources, définitions et bibliothèques à `mimir_core` ;
5. instanciez-le dans le code d'initialisation des runtimes ;
6. ajoutez-le à `RuntimeRouter::setRuntimes` et `setActivators` ;
7. documentez sa priorité par rapport aux backends existants ;
8. exposez ses capacités si l'API Lua de diagnostic doit les afficher.

Le routeur actuel a cinq emplacements explicites. Ajouter un sixième backend
nécessite donc de modifier les signatures et les structures internes du
routeur ; créer uniquement le dossier backend ne suffit pas.

## Exemple : refuser proprement `Add`

La signature réelle de `forwardLayer` est :

```cpp
bool MyRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    (void)training;
    if (!isInitialized()) return false;
    if (layer.type_enum != LayerType::Add) return false;
    if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
    if (inputs[0]->size() != inputs[1]->size()) return false;

    std::vector<float> result(inputs[0]->size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (*inputs[0])[i] + (*inputs[1])[i];
    }

    outputs.clear();
    outputs.push_back(std::move(result));
    return true;
}
```

Cet exemple utilise le même contrat élément par élément que le cas `Add` du
runtime CPU. Pour un backend accéléré, remplacez seulement le calcul après
validation ; le contrat de sortie reste identique.

## Tester le runtime

Un benchmark global ne démontre pas qu'un backend a exécuté l'opération visée.
Ajoutez un test C++ ciblé qui :

1. initialise le runtime ;
2. construit un `Layer` du type testé ;
3. compare sa sortie à une référence CPU ;
4. vérifie un cas refusé ;
5. vérifie la route et le runtime sélectionné ;
6. couvre la passe arrière si elle est annoncée.

Puis exécutez les tests runtime existants :

```bash
ctest --test-dir build --output-on-failure \
  -R 'RuntimeTest|AutogradTest.Numerical'
```

Les tests du planificateur constituent une suite distincte. Exécutez-les
séparément lorsque votre modification touche la planification, afin qu'un
échec de fusion ne soit pas confondu avec un échec du backend :

```bash
ctest --test-dir build --output-on-failure -R FrameworkTest.Planner
```

Pour observer le dispatch, appliquez les variables à un scénario ciblé dont
vous maîtrisez la taille. Par exemple, lancez d'abord le test de contrat ou le
script minimal associé à l'opération :

```bash
MIMIR_ACCEL_VERBOSE=1 MIMIR_RUNTIME_TRACE=1 \
  ctest --test-dir build --output-on-failure -R RuntimeTest.MathLinear
```

`benchmark_official.lua --safe` n'est pas un smoke test : il inclut plusieurs
tailles de Transformer jusqu'au profil Large et peut consommer plusieurs
gigaoctets.

Pour mesurer spécifiquement le forward NMS sans dataset :

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_nms.lua -- --quick
```

Ce résultat mesure le chemin réellement routé par `Model.forward`; il ne prouve
pas à lui seul qu'un kernel GPU natif a été utilisé.

## Critères de validation

- Les résultats correspondent au CPU dans la tolérance annoncée.
- Un cas non supporté retourne `false` sans sortie partielle.
- Les votes de support correspondent aux implémentations.
- La route conserve un fallback CPU fonctionnel.
- Les ressources sont libérées par `shutdown`.
- Les options CMake désactivées n'introduisent pas de dépendance de lien.

## Étapes suivantes

- [Ajouter une opération](05-Tuto-Ajouter-Op.md)
- [Développement des runtimes](../07-Devs/04-Runtime-Development.md)
- [Internals des runtimes GPU](../04-Architecture-Internals/21-GPU-Runtimes.md)
