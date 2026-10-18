# Atelier d’annotation Lumen

Application PHP locale, sans bibliothèque JavaScript, pour annoter récursivement un dataset. Chaque image conserve son JSON dans son propre dossier. Le texte de conditionnement principal est `captions.primary`, une description naturelle ; les entités, attributs, relations, boîtes, éclairage et composition complètent cette description.

## Lancer

PHP **8.1 ou supérieur**, avec l’extension `fileinfo`, est requis. Sur Ubuntu, le paquet `php-cli` fournit le lanceur PHP ; vérifiez `php -v` et `php -m` (`fileinfo` doit apparaître).

Depuis la racine du dépôt :

```sh
php -S 127.0.0.1:8080 -t annotator
```

Ouvrir **http://127.0.0.1:8080**, saisir le chemin du dataset et cliquer sur **Ouvrir / rescanner**. Le serveur doit pouvoir lire les images et écrire dans leurs dossiers et à la racine du dataset. L’application est prévue pour le serveur local PHP ; les requêtes provenant d’une adresse ou d’un hôte distant sont refusées.

## Annoter

1. Rédiger une phrase humaine dans la description principale. Les alternatives se saisissent une par ligne.
2. Glisser sur une zone libre de l’image pour créer une entité et sa boîte. Renseigner sa description et son rôle. L’identifiant automatique reste éditable.
3. Cliquer sur une boîte pour sélectionner l’entité, glisser son intérieur pour la déplacer, ou ses huit poignées pour la redimensionner. Les coordonnées enregistrées sont `[x_min, y_min, x_max, y_max]`, normalisées entre 0 et 1.
4. Pour remplacer la boîte d’une entité, utiliser **Sélectionner / dessiner** sur sa fiche. Pour dessiner un nouvel objet dans une boîte existante, utiliser **Dessiner une nouvelle boîte**.
5. Ajouter les attributs et les relations avec les sélecteurs d’entités, puis renseigner l’éclairage, la composition et les vérifications.
6. Cliquer sur **Sauvegarder** ou **Sauvegarder et suivante**. La sauvegarde requiert une description principale non vide et des références valides.

**Suppr** ou **Retour arrière**, quand le canvas a le focus, supprime uniquement la boîte sélectionnée. **Supprimer l’entité** supprime aussi ses attributs et ses relations, après confirmation. Renommer un identifiant met à jour ses références.

Le statut distingue une annotation existante, une modification non sauvegardée et une annotation sauvegardée. La navigation et la fermeture avertissent en présence de modifications. Les sauvegardes sont déclenchées par les boutons ; elles ne se font pas en arrière-plan pendant la saisie.

## Fichiers et reprise

- `portraits/photo.jpg` → `portraits/photo.json` (jamais `photo.jpg.json`).
- `.annotation_state.json` à la racine conserve le dossier, la dernière image visitée, son index **à partir de zéro**, les images annotées, la dernière sauvegarde et la date ISO 8601 avec fuseau.
- `.annotation_state.lock` coordonne les écritures d’historique et les sauvegardes concurrentes.
- Après réouverture du même dossier, l’outil reprend la dernière image visitée si elle est inachevée. Si elle est déjà annotée, il cherche la prochaine image non annotée, en bouclant dans la liste. Si tout est annoté, il conserve la dernière image.
- **Dernière annotation** revient à la dernière image sauvegardée. Le chemin du dernier dossier est mémorisé dans le navigateur ; l’ouverture reste explicite.
- Le compteur considère comme annotées les images ayant un JSON conforme et une description principale non vide. Les cases de vérification restent des indicateurs de qualité indépendants.

Les images JPG/JPEG, PNG, WebP et BMP sont scannées. L’affichage dépend des formats décodables par le navigateur. Les images originales ne sont jamais écrites. Les sous-dossiers symboliques ne sont pas parcourus ; les images symboliques ne sont acceptées que si leur cible reste dans le dataset, avec un JSON placé à côté du lien.

Un JSON existant invalide ou étranger au format est signalé et bloque l’édition de cette image. Les champs supplémentaires d’un JSON conforme sont conservés. Si `photo.jpg` et `photo.png` partagent un dossier, l’outil bloque leur annotation : il faut les renommer pour respecter la règle d’un JSON par image. Les liens symboliques de sortie sont refusés.

Les JSON sont écrits en UTF-8, formatés et remplacés atomiquement dans le même dossier. Une révision SHA-256 refuse de remplacer un fichier modifié depuis son chargement. Un onglet dont le dataset actif a changé doit rouvrir son dossier. Si le JSON est enregistré mais que l’historique échoue, un message explicite le signale ; un nouveau scan retrouve le JSON.

Mímir reconnaît ce format dans son loader natif. Si le JSON structuré est présent, `captions.primary` remplace la description historique pour les entraînements textuels, VAE conditionnés et DiT/Lumen ; sinon le loader conserve sa lecture historique, notamment du `.txt`. Les métadonnées complètes sont accessibles via `Mimir.Dataset.get(i).annotation`. Voir le [guide des datasets](../docs/02-User-Guide/03-Data.md#annotations-json-de-léditeur-lumen).

## Vérifier

```sh
node --check annotator/assets/app.js
python3 annotator/tests/smoke.py
```

Le test démarre son propre serveur PHP sur un port libre et crée uniquement un dataset temporaire, supprimé à la fin. Il couvre le scan récursif, les chemins Unicode, les sauvegardes et le rechargement, la reprise entre sessions, les révisions concurrentes, les références et boîtes invalides, les JSON étrangers, les collisions de noms, les liens symboliques, les traversées, les jetons CSRF, les hôtes distants et la conservation des images. `PHP_BIN` permet de sélectionner un autre binaire ; `PHP_ARGS` accepte ses options supplémentaires.

## Squelettes nommés

Chaque entité peut porter un squelette : **+ Squelette humain** propose tête, cou, épaules, bras, mains, hanche, jambes et pieds ; **+ Squelette personnalisé** démarre vide. Le nom du squelette et les noms des points sont libres. **Relié à** définit les segments. Utiliser **Placer**, puis cliquer sur l’image ; un point placé peut ensuite être déplacé par glissement. Les points non placés restent autorisés. Supprimer un point retire ses liaisons.

Le JSON conserve le squelette dans `entities[].skeleton`, sans modifier les descriptions, boîtes, attributs ou relations :

```json
{"name":"humain","points":[
  {"name":"hanche","position":[0.5,0.6],"parent":null},
  {"name":"tête","position":[0.5,0.2],"parent":"hanche"},
  {"name":"bras gauche","position":null,"parent":"hanche"}
]}
```

Les positions `[x,y]` sont normalisées sur l’image entière, entre 0 et 1. Les noms des points doivent être uniques dans chaque squelette. Les annotations précédentes sans squelette restent compatibles. Lumen exploite ces métadonnées dans son conditionnement textuel et la pondération spatiale de sa loss de diffusion existante, en entraînement comme en validation (voir le guide des datasets).

### Lier des éléments et personnaliser la structure

Dans chaque point, **Entités liées à ce point** permet de cocher plusieurs entités (par exemple une main liée à la personne et à un objet tenu). Les associations restent indépendantes des boîtes ; si le point et la boîte cible sont placés, un trait pointillé les relie. Les références suivent le renommage des entités et sont retirées à leur suppression.

**Liaisons libres entre points** ajoute autant de segments que nécessaire, avec un nom optionnel. Plusieurs liaisons par point, branches et cycles sont possibles, en complément du champ **Relié à** existant. Le modèle humain est entièrement éditable : ajouter, renommer, supprimer et repositionner ses points, effacer une position, modifier les liaisons et renommer le squelette. Le squelette personnalisé permet la même édition à partir de zéro, pour des personnes, animaux, objets ou structures abstraites.

Les associations sont stockées dans `points[].entities` (liste d’identifiants d’entités), les segments supplémentaires dans `skeleton.edges` (`from`, `to` : noms de points ; `label` : nom libre). Ces champs sont optionnels, pour conserver les anciens fichiers.

### Ajouter une queue

**+ Queue** ajoute un membre prêt à placer : base, deux segments intermédiaires et extrémité, reliés en chaîne. La base est attachée à `hanche`, `bassin` ou `pelvis` si ce point existe ; sinon, choisir son attache avec **Relié à**. Le bouton crée un squelette personnalisé si l’entité n’en possède pas encore. Plusieurs queues peuvent être ajoutées avec des noms distincts. Chaque point reste renommable, déplaçable, supprimable et associable aux entités ; **+ Point nommé** permet de détailler davantage la courbure. Le format JSON reste celui des points et liaisons déjà pris en charge par Lumen.

### Simuler la cohérence des attaches

Activer **Vérifier les liens squelette → entité** dans le panneau **Simulation des attaches**. Les attaches sont recalculées pendant l’édition et superposées avec quatre couleurs : vert **Plausible**, jaune **Ambigu**, rouge **À corriger**, gris **Non vérifiable**. Chaque résultat explique son raisonnement ; **Voir cette attache** sélectionne le squelette et met en évidence le point et la boîte cible.

La simulation mesure la distance du point au bord le plus proche de la boîte cible, en tenant compte des dimensions de l’image. La tolérance est réglable de 0 à 10 % de sa diagonale (3 % par défaut). Un point dans la boîte ou assez proche est plausible ; une boîte concurrente aussi proche rend le résultat ambigu. Une cible éloignée est signalée et une autre entité proche peut être proposée. **Remplacer par…** modifie uniquement le lien explicitement choisi ; il faut ensuite sauvegarder.

Les points racines (sans parent ni liaison libre entrante) vérifient aussi leur attache à l’entité propriétaire du squelette. Les autres points sans association explicite peuvent dépasser sa boîte, notamment les queues. Les cibles volontairement liées au même point et le propriétaire ne sont pas considérés comme des concurrents. Un point non placé, une boîte absente ou une image non chargée produit **Non vérifiable**.

Cette aide simule une cohérence géométrique ; elle ne reconnaît ni le contenu de l’image ni l’anatomie, et un lien plausible demande votre vérification visuelle. L’activation de la simulation et ses diagnostics ne modifient pas le JSON et ne valident pas automatiquement l’annotation.

Test des règles de simulation :

```sh
node annotator/tests/skeleton-links.cjs
```

### Contexte fourni aux modèles

Les bbox, rôles, squelettes et relations sauvegardés entrent dans le contexte textuel natif des modèles conditionnés. Les identifiants maintiennent les références entre entités et points. Ce contexte reste actif même si la pondération de loss spatiale vaut zéro. `Mimir.Model.conditioning_prompt(caption, annotation)` permet d’obtenir la représentation exacte pour la tokenisation ou la génération. Lumen et les entraînements VAE conditionnés refusent une séquence trop courte pour le contexte annoté ; augmenter respectivement `text_seq_len` et `seq_len` selon l’erreur. Les diagnostics du simulateur restent une aide d’édition et ne sont pas transmis comme des vérités au modèle.

### Nature des points du squelette

Le sélecteur « Nature du point » distingue la tête, les membres de support
(pieds), de saisie (mains), les articulations majeures (hanches) et les points
ordinaires. Les associations aux entités précisent les éléments tenus ou les
appuis. Le champ optionnel `points[].role` conserve ce choix ; les anciens
points sans rôle sont présentés selon leur nom.

Sur une tête, « Positionner le calque du visage » ajoute les deux yeux, les deux
oreilles, le nez et la bouche. Cliquez au centre du visage pour placer cet
ensemble, puis glissez chaque détail pour ajuster sa position. Le bouton permet
aussi de repositionner les six détails existants. Ces points sont enregistrés
comme enfants de la tête, avec le rôle `détail facial`.

Vérification : `node annotator/tests/skeleton-roles.cjs`.
