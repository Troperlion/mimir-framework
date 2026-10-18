# Tutoriels

Les tutoriels Mímir partent d'un résultat concret et expliquent progressivement
les mécanismes utilisés. Commencez par le parcours général si vous découvrez le
framework, puis choisissez un guide selon ce que vous souhaitez construire.

## Apprendre Mímir

- [Le framework en trois étapes](01-Cours-Framework-3-Etapes.md) propose un
  parcours débutant, intermédiaire et avancé.
- [Parcours complet du framework](06-Parcours-Complet-Framework.md) relie
  modèles, scripts, registre, runtimes et sérialisation.
- [Parcours par niveau](../01-Getting-Started/06-Learning-Paths.md) aide à
  sélectionner les pages adaptées à votre expérience.

## Modèles et scripts

- [Ajouter un modèle](02-Tuto-Ajouter-Modele.md)
- [Écrire un script Lua](03-Tuto-Coder-Script.md)
- [Du registre au checkpoint](08-Tuto-Registre-Pipeline-Checkpoint.md)
- [Créer et inspecter un package MPK](09-Tuto-Creer-MPK.md)
- [Rendre un run reproductible avec JSON et `env`](../02-User-Guide/08-Config-Driven-Scripting.md)
- [Valider VAEConv et son prior appris sans dataset](07-Tuto-VAEConv-Sans-Dataset.md)

## Moteur et runtimes

- [Modifier ou ajouter un runtime](04-Tuto-Modifier-Ou-Ajouter-Runtime.md)
- [Ajouter une opération](05-Tuto-Ajouter-Op.md)

> **Note**
> Les tutoriels VAEConv et registre/checkpoint n'utilisent aucun dataset. Les
> autres tutoriels indiquent leurs dépendances dans leurs prérequis.

## Avant de commencer

Compilez Mímir et validez le binaire :

```bash
./bin/mimir --help
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

Si cette commande échoue, revenez au
[guide d'installation](../01-Getting-Started/02-Installation.md) ou au
[smoketest](../01-Getting-Started/05-Smoketest.md).

## Étapes suivantes

Pour comprendre les API utilisées par les tutoriels, consultez la
[référence Lua](../03-API-Reference/00-API-Overview.md). Pour modifier le code
C++, poursuivez avec le [guide développeur](../07-Devs/00-INDEX.md).
