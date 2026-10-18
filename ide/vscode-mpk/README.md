# Mímir MPK pour VS Code

Cette extension locale fournit une aide IDE pour les sources pseudocode
`.mpk` :

- langage `mimir-mpk` et association avec l’extension `.mpk` ;
- coloration des déclarations `map`, `list`, `array`, de `.set` et `.append` ;
- commentaires `#`, chaînes, nombres, booléens et `null` ;
- snippets `mpk-package`, `mpk-map`, `mpk-list`, `mpk-set`, `mpk-append` et
  `mpk-node` ;
- complétion et documentation au survol pour les champs MPK et les LayerType.

Le fichier `mpk-stub.json` constitue la source déclarative de l’aide IDE.

## Développement

Pour lancer un hôte VS Code avec l’extension locale :

```bash
code --extensionDevelopmentPath="$PWD/ide/vscode-mpk"
```

Pour produire un VSIX si `vsce` est installé :

```bash
cd ide/vscode-mpk
vsce package
code --install-extension mimir-mpk-1.0.0.vsix
```

L’extension ne traite que les sources `.mpk`. Les fichiers `.mpk.bin` restent
des binaires opaques et ne doivent pas être associés à ce langage.
