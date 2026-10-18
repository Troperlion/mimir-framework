# Benchmark de compatibilite iCESugar-Pro

Ce test repond a deux questions distinctes :

1. Mimir et l'iCESugar-Pro peuvent-ils echanger des donnees sans corruption ?
2. Le temps de transfert et le calcul FPGA projete battent-ils le runtime CPU ?

Le bitstream renvoie chaque octet recu sur l'UART iCELink. Le programme
`mimir_fpga_benchmark` verifie les donnees bit a bit, mesure le debit utile, mesure
les vrais chemins `Conv2d` et `Linear` du `CpuRuntime`, puis calcule le speedup.

## Prerequis

Installer une toolchain ECP5 fournissant `yosys`, `nextpnr-ecp5` et `ecppack`.
L'[OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) les contient.
Il faut aussi un outil de programmation compatible, par exemple `icesprog` ou
les scripts `dapprog` du depot MuseLab.

Si `icesprog` affiche `iCELink open fail!` et que l'iCELink Pro `1d50:602b`
apparait en `root:root` dans `/dev/hidraw*`, installer une regle udev :

```bash
sudo sh -c 'cat > /etc/udev/rules.d/60-icesugar.rules <<EOF
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="602b", MODE="0660", GROUP="plugdev", TAG+="uaccess"
EOF'
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hidraw
```

Debrancher puis rebrancher la carte avant de relancer `icesprog`. L'utilisateur
doit appartenir au groupe `plugdev`. Un DAPLink `0d28:0204` avec un numero de
serie commencant par `0700` peut etre celui de l'Ext-Board ; brancher le cable
directement sur l'iCESugar-Pro, qui doit apparaitre avec un numero `0710`.

## Construire et charger le bitstream

```bash
make -C fpga/icesugar-pro/compat_echo
icesprog fpga/icesugar-pro/compat_echo/compat_echo.bit
```

Une autre methode de programmation MuseLab peut etre utilisee. Programmer le
bitstream interrompt toute logique actuellement chargee sur le FPGA.

Les contraintes proviennent de la carte iCESugar-Pro :

- horloge 25 MHz : `P6` ;
- FPGA vers iCELink : `B9` ;
- iCELink vers FPGA : `A9`.

## Construire le benchmark Mimir

La cible CMake `mimir_fpga_benchmark` est activee par defaut. Elle peut etre
retiree avec `MIMIR_BUILD_FPGA_BENCHMARK=OFF`.

Le test CPU et le modele de cout fonctionnent sans carte :

```bash
./bin/mimir_fpga_benchmark --iterations 15 --fpga-gmacs 2.0
```

Apres chargement du bitstream, utiliser le chemin stable de la carte :

```bash
./bin/mimir_fpga_benchmark \
  --serial /dev/serial/by-id/usb-MuseLab_DAPLink_CMSIS-DAP_*-if01 \
  --baud 115200 \
  --serial-bytes 4096 \
  --serial-rounds 10 \
  --iterations 15 \
  --fpga-gmacs 2.0
```

Le glob doit etre resolu par le shell vers un seul peripherique. Fermer au
prealable `picocom`, `screen` ou tout programme qui utilise le port.

## Interpretation

`Compatibilite materielle: PASS` signifie que toutes les trames sont revenues
sans erreur. Cela valide le chemin Mimir/hote vers iCELink vers FPGA et retour,
mais pas encore un noyau de reseau neuronal.

Pour chaque couche, le programme affiche :

- temps CPU median et p95 ;
- debit CPU en GMAC/s ;
- temps de calcul FPGA projete ;
- debit minimal du lien pour egaler le CPU ;
- temps total et speedup pour plusieurs liens.

Verdicts :

| Verdict | Signification |
| --- | --- |
| `RENTABLE` | Speedup superieur ou egal a 1,20 |
| `MARGINAL` | Speedup entre 1,00 et 1,20 |
| `NON` | FPGA projete plus lent que le CPU |

Le calcul suppose des poids INT8 deja residents sur le FPGA. Il inclut le
transfert de l'entree et de la sortie, mais pas le chargement initial des poids.
Il est donc favorable au FPGA. Si le resultat reste `NON`, un accelerateur couche
par couche ne vaut pas son cout pour cette forme.

Un resultat `NON` n'exclut pas un sous-graphe resident. Dans ce cas, seules
l'entree du premier layer et la sortie du dernier traversent le lien. Il faut
relancer le calcul avec les tailles de frontiere du sous-graphe avant de conclure.

## Limites du test

- L'echo UART mesure le transport, pas les 28 DSP ni la SDRAM.
- La FIFO RTL absorbe les variations du pont USB-CDC ; un tampon d'un seul octet
  perd des donnees sur les trames soutenues de 4096 octets.
- `--fpga-gmacs` est une hypothese jusqu'a l'implementation du noyau INT8.
- Les formes mesurees sont representatives des configurations du depot, mais ne
  remplacent pas un profil couche par couche du modele final.
- La quantification et son impact sur la qualite ne sont pas mesures ici.
