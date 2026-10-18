# Bitstream runtime Mimir pour iCESugar-Pro

Ce bitstream est le premier protocole de calcul reconnu par `FpgaRuntime`. Il
remplace le bitstream echo pour l'utilisation du runtime, sans modifier le banc
de diagnostic `compat_echo`.

Fonctions exposees :

- handshake binaire versionne `1.0` ;
- masque de capacites ;
- produit scalaire signe INT8 avec accumulation INT32 ;
- cache de poids INT8 resident avec commandes `LOAD` et `EXEC` ;
- matrice INT8 residente `8x64` avec huit accumulateurs en parallele ;
- vecteurs de 1 a 64 elements.

Le runtime exige simultanement l'identite USB `1d50:602b`, une serie commencant
par `0710`, un tty accessible et une reponse de capacites valide. L'ancien echo,
une autre carte ou un protocole de version majeure differente sont refuses.

## Construire et programmer

```bash
make -C fpga/icesugar-pro/mimir_runtime
cp fpga/icesugar-pro/mimir_runtime/mimir_runtime.bit /run/media/$USER/iCELink/
sync
```

La synthese du noyau matriciel mesure une Fmax de 64,35 MHz pour l'horloge
contrainte a 25 MHz. Elle utilise 11 des 28 blocs `MULT18X18D` disponibles.

## Valider sur la carte

```bash
./bin/mimir_fpga_benchmark \
  --serial /dev/serial/by-id/usb-MuseLab_DAPLink_CMSIS-DAP_0710*-if01 \
  --runtime-protocol \
  --iterations 3
```

Le test valide le handshake puis compare les accumulateurs FPGA et CPU pour des
vecteurs de 1, 7, 32 et 64 elements. Il charge ensuite 64 poids une seule fois,
execute dix entrees distinctes et compare chaque resultat au CPU. Le glob du tty
doit etre resolu par le shell vers un seul chemin.

Mesure reelle a 115200 bauds pour 64 elements :

| Mode | Mediane |
| --- | ---: |
| `DOT8`, entree et poids transmis | 12,97 ms |
| `EXEC`, poids deja residents | 7,05 ms |

La residence apporte `1,84x` sur ce transport. Elle ne rend toutefois pas un
vecteur isole rentable face au CPU ; le vote de couche FPGA reste donc desactive.

Pour une matrice residente `8x64`, les huit lignes sont calculees en parallele :

| Mode | Mediane |
| --- | ---: |
| Huit commandes `DOT8` | 103,99 ms |
| Une commande `MVEC`, huit sorties | 10,01 ms |

Le gain mesure est de `10,39x` face aux huit commandes separees. Les dix vecteurs
du benchmark et leurs 80 sorties ont ete compares exactement aux accumulateurs
CPU. Cette mesure confirme que la bonne direction est d'augmenter le nombre de
sorties traitees par transfert, et non d'offloader des produits scalaires isoles.

Ce noyau prouve le calcul sur le FPGA, mais n'est pas vote comme une couche
`Linear` : le contrat actuel de `Linear` est FP32 et le transport UART rendrait
le streaming couche par couche plus lent que CPU. Le prochain noyau de production
doit conserver une matrice de poids et un sous-graphe complet dans la SDRAM du
FPGA, puis amortir le transport sur plusieurs sorties ou plusieurs couches.