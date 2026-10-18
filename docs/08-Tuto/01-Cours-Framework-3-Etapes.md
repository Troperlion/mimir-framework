# Apprendre le framework en trois étapes

Apprendre a utiliser Mimir de facon progressive, du premier modele jusqu'a l'optimisation et l'extension du framework.

**Public concerné :** - Debutants (college, lycee)
- Etudiants (ingenierie, prof, autodidactes techniques)
- Niveau avance (fac, scientifique, dev senior)

> **Prérequis**
>
> 1. Compiler Mimir (voir [docs/01-Getting-Started/02-Installation.md](../01-Getting-Started/02-Installation.md)).
> 2. Verifier l'environnement avec le smoketest (voir [docs/01-Getting-Started/05-Smoketest.md](../01-Getting-Started/05-Smoketest.md)).
> 3. Savoir lancer une commande terminal simple.

- creer un script de base,
- lancer entrainement et inference,
- diagnostiquer un probleme de performance ou de memoire,
- choisir la documentation adaptee a votre niveau.

## Sur cette page

- [Parcours numerote 1 a 5](#parcours-numerote-1-a-5)
- [Etape 1 - Debutants (college, lycee)](#etape-1---debutants-college-lycee)
- [Etape 2 - Etudiants (ingenierie, prof, etc.)](#etape-2---etudiants-ingenierie-prof-etc)
- [Etape 3 - Avance (fac, scientifique, dev senior)](#etape-3---avance-fac-scientifique-dev-senior)
- [Conseils de progression](#conseils-de-progression)
- [Suite logique](#suite-logique)
- [Étapes suivantes](#étapes-suivantes)

## Parcours numerote 1 a 5

1. Installer et verifier l'environnement
2. Lancer un premier modele qui fonctionne
3. Distinguer construction, entraînement et inférence
4. Sauvegarder, recharger et comparer les resultats
5. Mesurer, optimiser, puis valider la non-regression

Ce parcours est compatible avec les 3 niveaux ci-dessous: debutant, etudiant, avance.

---

## Etape 1 - Debutants (college, lycee)

### Mission

Comprendre ce qu'est un modele, lancer un script qui marche, et lire une sortie.

### Notions a retenir

- Un modele prend une entree et produit une sortie.
- Dans Mimir, l'entree par defaut est souvent `__input__` et la sortie principale `x`.
- On suit un cycle: creation -> allocation -> execution.

### Parcours recommande (45 a 90 min)

1. Lire [docs/01-Getting-Started/00-GET-STARTED.md](../01-Getting-Started/00-GET-STARTED.md).
2. Faire le quick start [docs/01-Getting-Started/01-Quick-Start.md](../01-Getting-Started/01-Quick-Start.md).
3. Comprendre les concepts de base [docs/02-User-Guide/01-Core-Concepts.md](../02-User-Guide/01-Core-Concepts.md).

### Exercices

1. Construire le plus petit modèle enregistré :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch basic_mlp --no-train
```

2. Identifier dans les logs:
- si le modele est cree,
- si l'allocation memoire est faite,
- si un forward passe sans erreur.

3. Modifier un parametre simple (taille, batch, ou seed) et relancer.

### Validation de fin d'etape

Vous validez l'etape si:

- vous lancez un script sans crash,
- vous expliquez en une phrase le role de `forward`,
- vous savez ou regarder en cas d'erreur de lancement.

---

## Etape 2 - Etudiants (ingenierie, prof, etc.)

### Mission

Construire et sérialiser un modèle, puis comprendre séparément ce qu'exige un
véritable entraînement.

### Notions a retenir

- Le cycle moderne : `create` construit le graphe, `build` reste un no-op de
  compatibilité, puis viennent `allocate`, `init`, `forward` et `backward`.
- Le role du dataset, des hyperparametres et des checkpoints.
- La difference entre entrainement et inference.

### Parcours recommande (1 a 2 jours)

1. Lifecycle modele: [docs/02-User-Guide/02-Model-Lifecycle.md](../02-User-Guide/02-Model-Lifecycle.md).
2. Donnees: [docs/02-User-Guide/03-Data.md](../02-User-Guide/03-Data.md).
3. Entrainement: [docs/02-User-Guide/04-Training.md](../02-User-Guide/04-Training.md).
4. Inference: [docs/02-User-Guide/05-Inference.md](../02-User-Guide/05-Inference.md).
5. Checkpoints: [docs/02-User-Guide/08-Checkpoints.md](../02-User-Guide/08-Checkpoints.md).

### Exercices

1. Exécuter le parcours registre vers checkpoint, qui ne demande aucun dataset :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch basic_mlp --no-train \
  --save /tmp/mimir_course_basic_mlp.safetensors
```

2. Inspecter le checkpoint avec `scripts/tools/analyze_model.lua`.
3. Lire l'aide d'un script d'entraînement adapté à votre architecture.
4. Si vous disposez ensuite des données requises, faire un run court et
   documenter dataset, seed, configuration et métrique.

### Validation de fin d'etape

Vous validez l'etape si:

- vous reproduisez un cycle build, sauvegarde et inspection,
- vous savez expliquer la difference train/inference,
- vous savez identifier les prérequis supplémentaires d'un entraînement réel.

---

## Etape 3 - Avance (fac, scientifique, dev senior)

### Mission

Analyser le framework en profondeur, optimiser les performances, et etendre les composants.

### Notions a retenir

- API et contrat de scripting inter-langages.
- Architecture interne du moteur d'execution.
- Gestion memoire, runtime allocator, et stabilite numerique.

### Parcours recommande (2 a 5 jours)

1. API overview: [docs/03-API-Reference/00-API-Overview.md](../03-API-Reference/00-API-Overview.md).
2. Memory API: [docs/03-API-Reference/14-Memory.md](../03-API-Reference/14-Memory.md).
3. Internals engine: [docs/04-Architecture-Internals/01-Engine-Overview.md](../04-Architecture-Internals/01-Engine-Overview.md).
4. RuntimeAllocator/scratchpads: [docs/04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md](../04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md).
5. Debug/perf: [docs/05-Advanced/01-Performance.md](../05-Advanced/01-Performance.md) et [docs/05-Advanced/02-Debugging.md](../05-Advanced/02-Debugging.md).
6. Dev guide: [docs/07-Devs/00-INDEX.md](../07-Devs/00-INDEX.md).

### Exercices

1. Mesurer d'abord les tests runtime ciblés :

```bash
ctest --test-dir build --output-on-failure \
  -R 'RuntimeTest.MathLinear|RuntimeTest.MathConv2d|RuntimeTest.MathNorms'
```

2. Proposer une optimisation (memoire, layout, batch, precision dtype).
3. Implementer un changement minimal et mesurer avant/apres.
4. Ecrire une note technique: objectif, methode, resultats, limites.

> **Attention**
> `benchmark_official.lua --safe` inclut actuellement les profils Warmup,
> Small, Medium et Large. Même avec `--iters 1`, il peut approcher la limite
> mémoire de 10 Go. Ne l'utilisez qu'après avoir vérifié sa configuration dans
> le script et la mémoire disponible.

### Validation de fin d'etape

Vous validez l'etape si:

- vous produisez une mesure avant/apres argumentee,
- vous reliez votre resultat a des choix runtime/memoire,
- vous identifiez un risque de regression et un test de non-regression.

---

## Conseils de progression

1. Ne sautez pas les étapes : la base accélère réellement le niveau avancé.
2. Gardez un journal d'expériences avec commande, config et résultat.
3. Utilisez les checkpoints pour comparer proprement les changements.
4. Si un run échoue, réduisez l'expérience et isolez le problème.

## Suite logique

- Approfondir VAEText: [docs/02-User-Guide/11-VAEText.md](../02-User-Guide/11-VAEText.md)
- Approfondir Transformer/GPT: [docs/02-User-Guide/12-Transformer-GPT.md](../02-User-Guide/12-Transformer-GPT.md)
- Approfondir diffusion: [docs/02-User-Guide/13-Diffusion.md](../02-User-Guide/13-Diffusion.md)

## Étapes suivantes

- [Page précédente : Tutoriels](00-INDEX.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tuto - Ajouter un modele](02-Tuto-Ajouter-Modele.md)
