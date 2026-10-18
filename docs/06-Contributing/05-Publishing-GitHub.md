# Préparer une publication GitHub

Cette procédure transforme un workspace de recherche en changement publiable
sans inclure datasets, checkpoints, secrets, chemins machine ou résultats
générés. Elle s'applique depuis la racine du dépôt.

## Avant de publier

```bash
git status --short
git diff --check
rg -n '/home/|/run/media/|BEGIN .*PRIVATE KEY|token[=:]' \
  README.md docs configs scripts src Tests fpga
```

Examinez chaque résultat : un chemin générique comme `/home/user/project` dans
un tutoriel est acceptable; un nom d'utilisateur, UUID de disque ou dataset
privé ne l'est pas.

## Ce qui appartient au dépôt

- sources C++, Lua, HDL et shaders ;
- tests déterministes et petites fixtures indispensables ;
- configurations portables avec chemins relatifs ;
- documentation source-faithful ;
- images d'interface intentionnelles et légères ;
- contraintes FPGA et scripts de construction.

## Ce qui reste local

- `build/`, `bin/`, logs et caches ;
- `dataset*/`, `checkpoint*/` et poids ;
- CSV de métriques, rapports HTML/TXT/Markdown générés et snapshots ;
- sorties de synthèse FPGA (`.bit`, `.svf`, `.config`, netlists JSON) ;
- paramètres d'interface utilisateur persistants ;
- archives de travail et rapports matériels.

## Validation minimale

```bash
jq empty configs/*.json
./scripts/tools/verify_api_sync.sh
./scripts/tools/verify_docs.py
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Si un backend optionnel n'est pas disponible, indiquez clairement la matrice
testée dans la pull request. Ne transformez pas un test CPU en revendication de
support matériel.

## Pull request reproductible

```bash
git switch -c documentation-source-aligned-publication
git add -A
git commit -m "docs: align project documentation and publication workflow"
git push -u origin documentation-source-aligned-publication
gh pr create --fill
```

Le résumé de PR doit séparer documentation, fonctionnalités, nettoyage et
preuves de validation. Ajoutez les limitations restantes dans le corps de la
PR plutôt que de les cacher.
