# Style de la documentation Mímir

La documentation Mímir suit une organisation inspirée de Laravel : navigation
stable par grands domaines, contenu progressif, titres descriptifs, exemples
proches du texte qu'ils illustrent et liens vers l'étape logique suivante.

Cette inspiration concerne l'expérience de lecture. Le contenu, les commandes
et les conventions restent propres à Mímir.

## Organisation générale

La navigation principale utilise les groupes suivants :

1. Prologue
2. Bien démarrer
3. Concepts d'architecture
4. Les bases
5. Modèles
6. Référence de l'API Lua
7. Runtimes et matériel
8. Aller plus loin
9. Développement et extension
10. Tutoriels

Une page appartient à un domaine principal. Les liens vers les autres domaines
sont placés dans le texte ou dans « Étapes suivantes », sans recopier tout
l'index général.

## Structure d'une page

Une page commence par un titre de niveau 1 et un court paragraphe qui répond à
deux questions :

- de quoi traite cette page ;
- dans quel cas le lecteur doit l'utiliser.

Elle continue avec des sections de niveau 2 orientées vers le sujet :

```markdown
# Entraînement

Cette page explique...

## Configurer un entraînement

## Lancer la boucle

### Reprendre un checkpoint

## Gérer les erreurs

## Étapes suivantes
```

Les blocs génériques « Pour qui », « Objectif », « Avant de commencer » et
« Résultat attendu » ne sont plus obligatoires. Une page de tutoriel peut
cependant conserver des prérequis et un résultat attendu lorsque cela aide à
exécuter les étapes.

## Table des matières

Les pages longues commencent par « Sur cette page », juste après
l'introduction. Cette liste contient uniquement les sections principales et,
si nécessaire, leurs sous-sections importantes.

```markdown
## Sur cette page

- [Créer un modèle](#créer-un-modèle)
- [Lancer une inférence](#lancer-une-inférence)
- [Gérer les erreurs](#gérer-les-erreurs)
```

## Écriture progressive

Présenter d'abord le chemin le plus courant. Ajouter ensuite les variantes, les
options avancées et les détails d'implémentation.

- Définir un terme lors de sa première apparition.
- Utiliser des phrases courtes et factuelles.
- Préférer « vous » au mélange entre « tu » et « vous ».
- Éviter les titres décoratifs, les emojis et les mentions comme « nouveau »
  qui deviennent rapidement obsolètes.
- Ne pas présenter une fonctionnalité expérimentale comme stable.
- Indiquer la version seulement lorsqu'elle change réellement le comportement.

## Exemples exécutables

Une commande est donnée depuis la racine du dépôt, sauf indication contraire.
Les arguments destinés au script Lua sont séparés de ceux de Mímir par `--` :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_conv --params
```

Après un exemple, expliquer brièvement :

1. ce que fait la commande ;
2. quel résultat doit être observé ;
3. quelles erreurs courantes peuvent survenir.

Un exemple incomplet doit porter la mention « schéma », « pseudo-code » ou
« extrait ». Il ne doit pas ressembler à une commande directement exécutable.

## Notes et avertissements

Utiliser des blocs courts et nommés :

```markdown
> **Note**
> Information complémentaire qui n'empêche pas l'exécution.

> **Attention**
> Condition pouvant invalider le résultat ou provoquer une perte de travail.

> **Limitation**
> Fonctionnalité absente, partielle ou expérimentale.
```

Les avertissements ne remplacent pas une explication. Ils mettent en évidence
une contrainte déjà décrite par la page.

## Référence technique

Une page de référence privilégie :

- la signature exacte ;
- les paramètres, leurs types et leurs valeurs par défaut ;
- les préconditions ;
- la forme des entrées et sorties ;
- les erreurs observables ;
- un exemple minimal ;
- le fichier source qui fait autorité.

Les API obsolètes sont marquées « dépréciées » et ne sont jamais recommandées
dans les guides de démarrage.

## Tutoriels

Un tutoriel suit un résultat concret :

1. prérequis ;
2. résultat final ;
3. étapes numérotées ;
4. vérification après chaque étape importante ;
5. explication du fonctionnement ;
6. problèmes fréquents ;
7. étapes suivantes.

Le tutoriel indique explicitement s'il utilise un dataset, un checkpoint
externe ou un backend matériel particulier.

## Conventions Mímir

| Nom | Convention |
| --- | --- |
| `__input__` | Entrée par défaut |
| `text_ids` | Entrée d'identifiants textuels |
| `x` | Sortie principale |

Les noms d'API, de fichiers, de variables et de tenseurs sont écrits entre
accents graves. Les noms de concepts ordinaires restent en texte normal.

## Validation avant publication

Avant de considérer une page terminée :

- vérifier les liens locaux ;
- exécuter les commandes présentées comme copiables ;
- vérifier les options contre `--help` ou le code ;
- confirmer les noms d'architectures dans le registre ;
- confirmer les capacités runtime dans l'implémentation ;
- lancer `git diff --check`.

## Étapes suivantes

Consultez l'[index général](00-INDEX.md) pour voir comment cette charte se
traduit dans la navigation, puis utilisez une page existante du même domaine
comme modèle.
