# Ajouter une opération

Ce tutoriel utilise l'opération existante `Multiply` comme trace de référence.
Il montre tous les endroits réellement impliqués dans une opération
entraînable, sans prétendre qu'un type fictif est déjà disponible.

## Sources de vérité

- `src/LayerTypes.hpp`
- `src/Layers.hpp`
- `src/runtimes/cpu/RuntimeLayerDispatch.hpp`
- `src/runtimes/LayerOps.hpp` et `src/runtimes/LayerOps.cpp`
- `src/runtimes/cpu/LayerOpsExt.hpp`
- `src/runtimes/AbstractRuntime.hpp`
- `Tests/test_autograd_numerical.cpp`
- `Tests/CMakeLists.txt`

## Étape 1 — Écrire le contrat mathématique

Pour `Multiply`, avec deux entrées de même taille :

\[
y_i = a_i b_i
\]

La passe arrière est :

\[
\frac{\partial L}{\partial a_i}
=
\frac{\partial L}{\partial y_i} b_i,
\qquad
\frac{\partial L}{\partial b_i}
=
\frac{\partial L}{\partial y_i} a_i.
\]

Avant de coder une nouvelle opération, fixez :

- le nombre d'entrées et de sorties ;
- les formes acceptées et les règles de broadcasting éventuelles ;
- le comportement pour une entrée vide ;
- les paramètres entraînables ;
- la dérivée ou le statut « inférence uniquement ».

## Étape 2 — Déclarer le type

`src/LayerTypes.hpp` contient trois éléments à garder synchronisés :

1. la valeur dans `enum class LayerType` ;
2. la table texte vers enum de `LayerRegistry` ;
3. la conversion enum vers texte et les familles de types supportés.

`Model::push` reçoit un nom de type sous forme de chaîne. Si
`LayerRegistry::string_to_type` ne connaît pas cette chaîne, le nouveau layer ne
sera pas routé correctement.

Ajoutez dans `src/Layers.hpp` uniquement les attributs réellement nécessaires
à l'opération. Une opération sans état, comme `Multiply`, n'a pas besoin de
champ supplémentaire.

## Étape 3 — Implémenter la passe avant CPU

La référence CPU se trouve dans le grand `switch` de
`RuntimeLayerDispatch::cpu_forward_layer`. Le cas actuel de `Multiply` vérifie
la seconde entrée, crée une sortie et appelle
`LayerOps::multiply_forward`.

Pour une nouvelle opération :

1. refusez les entrées invalides avec `false` ;
2. redimensionnez `outputs` selon le nombre de sorties contractuel ;
3. effectuez le calcul complet ;
4. retournez `true` seulement après production de la sortie.

Le runtime CPU est le dernier fallback. Une opération utilisable par un modèle
doit donc y être implémentée, sauf si elle est explicitement limitée à un
backend particulier.

## Étape 4 — Implémenter la passe arrière

Le même fichier contient `RuntimeLayerDispatch::cpu_backward_layer`. Le cas
`Multiply` :

- exige deux entrées et un gradient de sortie ;
- vérifie que les trois vecteurs ont la même taille ;
- produit exactement deux gradients d'entrée ;
- applique les deux dérivées données plus haut.

Si le layer possède des paramètres, accumulez aussi `grad_weights` ou
`grad_bias` selon la disposition décrite par `Layer`. Ne mettez pas à jour les
poids dans `backwardLayer` : cette responsabilité appartient à
`optimizerStep`.

## Étape 5 — Mettre à jour les votes runtime

Vérifiez `supportsForwardLayerType` et `supportsBackwardLayerType` pour chaque
backend concerné. Un vote positif sans implémentation peut créer une route
inutile ou trompeuse. Un vote négatif empêche le routeur de proposer
l'opération au backend.

Les kernels CUDA, ROCm, Vulkan et OpenCL sont optionnels. Ajoutez-les seulement
après avoir établi la référence CPU et les tests numériques.

## Étape 6 — Utiliser l'opération dans un modèle

Une opération binaire existante se câble par noms de tenseurs :

```cpp
model.push("example/multiply", "Multiply", 0);
if (auto* layer = model.getLayerByName("example/multiply")) {
    layer->inputs = {"example/a", "example/b"};
    layer->output = "x";
}
```

`params_count` vaut zéro parce que `Multiply` ne possède aucun poids.

`Mimir.Layers` n'est pas le chemin principal d'exécution des graphes de modèle.
Le chemin réel passe par les `Layer` construits en C++, puis par
`RuntimeRouter`.

## Étape 7 — Tester mathématiquement

Ajoutez trois niveaux de tests :

1. passe avant sur de petits vecteurs connus ;
2. passe arrière analytique ;
3. vérification par différences finies.

Pour une composante \(x_i\), une approximation centrée est :

\[
\frac{\partial L}{\partial x_i}
\approx
\frac{L(x_i+\varepsilon)-L(x_i-\varepsilon)}{2\varepsilon}.
\]

Choisissez une tolérance compatible avec `float32` et documentez `epsilon`.
Les tests existants dans `Tests/test_autograd_numerical.cpp` montrent le
pattern attendu.

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure \
  -R 'Autograd|Math|Runtime'
```

## Checklist complète

- Type présent dans toutes les conversions de `LayerTypes.hpp`.
- Attributs du layer initialisés avec une valeur sûre.
- Passe avant CPU correcte.
- Passe arrière correcte ou limitation explicitement documentée.
- Votes runtime cohérents.
- Modèle de test câblé avec les bons noms de tenseurs.
- Test nominal, formes invalides et différences finies.
- Documentation de la forme, du dtype et des paramètres.

## Étapes suivantes

- [Modifier ou ajouter un runtime](04-Tuto-Modifier-Ou-Ajouter-Runtime.md)
- [Internals des layers](../04-Architecture-Internals/14-Layers-And-Ops.md)
- [Autograd et gradients](../04-Architecture-Internals/13-Autograd-Gradients.md)
