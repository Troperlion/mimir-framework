# Métriques des rapports `show-graph.lua`

`scripts/tools/show-graph.lua` transforme un ou plusieurs CSV d'entraînement en
rapport HTML et, avec `--out-text`, en rapport Markdown ou texte brut. Cette page
définit les métriques affichées par le rapport à partir de leur implémentation
actuelle dans `src/Model.cpp`.

```bash
./bin/mimir --lua scripts/tools/show-graph.lua -- \
  --csv checkpoint/run/loss_history.csv \
  --checkpoint-dir checkpoint/run \
  --out graph_report.html \
  --out-text graph_report.md \
  --no-interactive
```

Le rapport lit les valeurs déjà enregistrées dans le CSV. Il ne recalcule pas
les métriques du modèle. Les sections statistiques, corrélations et écarts de
validation sont en revanche calculées par `show-graph.lua`.

## Notations et moments

On note $p_i$ la prédiction ou reconstruction, $t_i$ la cible et $N$ le
nombre de valeurs comparées. Mímir utilise les moments population :

\[
\mu_x=\frac{1}{N}\sum_{i=1}^{N}x_i
\]

\[
\sigma_x^2=\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu_x)^2
\]

Le troisième moment standardisé, ou asymétrie, vaut :

\[
\gamma_x=
\frac{\frac{1}{N}\sum_i(x_i-\mu_x)^3}{\sigma_x^3}
\]

Quand l'écart-type est inférieur ou égal à `1e-12`, l'asymétrie est fixée à
zéro. Voir `compute_moments()` et `compute_moments_prefix()` dans
`src/Model.cpp`.

## Reconstruction

Le champ CSV historique `mse` représente la métrique de reconstruction. Selon
`model_config.recon_loss`, il ne s'agit donc pas nécessairement d'une MSE. En
posant $d_i=p_i-t_i$, les variantes natives sont les suivantes.

### MSE

\[
L_{\mathrm{MSE}}=\frac{1}{N}\sum_i d_i^2
\]

### L1 ou MAE

\[
L_{\mathrm{L1}}=\frac{1}{N}\sum_i |d_i|
\]

### Charbonnier

\[
L_{\mathrm{Charbonnier}}=
\frac{1}{N}\sum_i\sqrt{d_i^2+\varepsilon^2}
\]

Le paramètre \(\varepsilon\) vient de `charbonnier_eps` et est borné au minimum
à `1e-12`.

### Huber ou Smooth L1

\[
\ell_\delta(d)=
\begin{cases}
\frac{1}{2}d^2,& |d|\leq\delta\\
\delta\left(|d|-\frac{1}{2}\delta\right),& |d|>\delta
\end{cases}
\]

\[
L_{\mathrm{Huber}}=\frac{1}{N}\sum_i\ell_\delta(d_i)
\]

`huber_delta`, `smoothl1_delta` ou `smoothl1_beta` fournit \(\delta\).

### Gaussian NLL

Pour un écart-type fixe \(\sigma\) fourni par `nll_sigma` ou
`gaussian_nll_sigma` :

\[
L_{\mathrm{NLL}}=\frac{1}{N}\sum_i
\frac{1}{2}\left(\frac{d_i^2}{\sigma^2}+\log(\sigma^2)\right)
\]

## KL Divergence

Pour un VAE dont la sortie latente contient les moyennes \(\mu_j\) et les
log-variances \(\ell_j=\log\sigma_j^2\) :

\[
D_{\mathrm{KL}}=
\frac{1}{2D}\sum_{j=1}^{D}
\left(\mu_j^2+\exp(\ell_j)-1-\ell_j\right)
\]

Les log-variances sont d'abord bornées par `logvar_clip_min` et
`logvar_clip_max`. Pour le diagnostic générique entre deux distributions
gaussiennes estimées :

\[
D_{\mathrm{KL}}=
\frac{1}{2}\left[
\log\left(\frac{\sigma_p^2}{\sigma_t^2}\right)
+\frac{\sigma_t^2+(\mu_t-\mu_p)^2}{\sigma_p^2}-1
\right]
\]

Le diagnostic générique affiché est borné inférieurement à zéro.

## Loss totale d'un VAE

La forme principale est :

\[
L_{\mathrm{total}}=mL_{\mathrm{reconstruction}}
+\beta_{\mathrm{eff}}D_{\mathrm{KL}}
\]

La reconstruction peut recevoir les composantes optionnelles SSIM, spectrale
et perceptuelle :

\[
L_{\mathrm{reconstruction}}=L_{\mathrm{pixel}}
+\lambda_{\mathrm{SSIM}}L_{\mathrm{SSIM}}
+\lambda_{\mathrm{spectral}}L_{\mathrm{spectral}}
+\lambda_{\mathrm{perceptual}}L_{\mathrm{perceptual}}
\]

Lorsque les marqueurs sont actifs :

\[
m=\operatorname{clamp}\left(
1+\lambda_W W+\lambda_T(1-r),\ 0.1,\ m_{\max}
\right)
\]

Sinon, $m=1$. Ici, $W$ est la métrique Wasserstein et $r$ la corrélation
de Pearson décrites ci-dessous.

## Wasserstein

Mímir utilise la distance $W_2$ entre les approximations gaussiennes
unidimensionnelles de la prédiction et de la cible :

\[
W_2=\sqrt{(\mu_t-\mu_p)^2+
\left(\sqrt{\sigma_t^2}-\sqrt{\sigma_p^2}\right)^2}
\]

## Entropy Delta

`entropy_diff`, affiché comme « Entropy Δ », est une différence d'entropie sous
approximation gaussienne :

\[
\Delta H=\frac{1}{2}
\left[\log(\sigma_p^2)-\log(\sigma_t^2)\right]
=\frac{1}{2}\log\left(\frac{\sigma_p^2}{\sigma_t^2}\right)
\]

Chaque variance est remplacée par \(\max(\sigma^2,10^{-12})\) avant le
logarithme. Une valeur positive indique une variance de prédiction supérieure ;
une valeur négative indique une variance inférieure.

## Moment Mismatch

\[
M_{\mathrm{moment}}=|\gamma_p-\gamma_t|
\]

Une valeur proche de zéro indique des asymétries de distribution proches.

## Spatial Coherence

Pour le vecteur aplati :

\[
A(x)=\frac{1}{N-1}\sum_{i=2}^{N}|x_i-x_{i-1}|
\]

\[
C_{\mathrm{spatial}}=|A(p)-A(t)|
\]

Malgré son nom, cette implémentation compare actuellement les éléments
adjacents du tenseur aplati ; elle ne parcourt pas séparément les voisinages
horizontaux et verticaux d'une image.

## Temporal Consistency

Cette métrique est la corrélation de Pearson :

\[
r=\frac{\sum_i(p_i-\mu_p)(t_i-\mu_t)}
{\sqrt{\sum_i(p_i-\mu_p)^2}\sqrt{\sum_i(t_i-\mu_t)^2}}
\]

Le résultat est borné dans \([-1,1]\). Mímir retourne zéro si le dénominateur
est inférieur ou égal à `1e-18`.

## Valeurs de pilotage affichées

Les champs suivants sont lus directement depuis le CSV :

- `learning_rate` est le taux d'apprentissage effectif du step ; sa formule
  dépend du scheduler choisi ;
- `timestep` est le pas échantillonné, notamment pour un modèle de diffusion ;
- `opt_eps` est l'epsilon numérique de l'optimiseur. Pour Adam/AdamW, il apparaît
  dans une mise à jour de la forme
  \(\theta_s=\theta_{s-1}-\eta\widehat m_s/(\sqrt{\widehat v_s}+\varepsilon)\) ;
- `step`, `opt_step` et `epoch` sont des index, pas des métriques dérivées.

## Validation

La loss de validation est la moyenne sur les $B$ éléments évalués :

\[
L_{\mathrm{val}}=\frac{1}{B}\sum_{b=1}^{B}L(x_b;\theta)
\]

`val_mse` dépend du producteur du CSV : MSE de reconstruction pour un modèle
classique, MSE en espace bruit pour DDPM, ou valeur secondaire spécifique au
VAE. Le rapport ne change pas sa signification.

L'écart validation-entraînement est :

\[
G_s=L_{\mathrm{val},s}-L_{\mathrm{train},s^*}
\]

où $s^*$ désigne le step d'entraînement disponible le plus proche du step de
validation. Un écart positif signifie que la loss de validation est supérieure.

## Calculs propres au rapport

### Statistiques

Pour chaque colonne, le tableau affiche :

\[
x_{\min}=\min_i x_i,\qquad x_{\max}=\max_i x_i
\]

\[
\bar{x}=\frac{1}{N}\sum_i x_i,\qquad
\sigma_x=\sqrt{\frac{1}{N}\sum_i(x_i-\bar{x})^2}
\]

L'écart-type est un écart-type population, divisé par $N$.

### Matrice de corrélation

Chaque cellule applique Pearson aux deux colonnes concernées :

\[
r_{XY}=\frac{\sum_i(X_i-\bar X)(Y_i-\bar Y)}
{\sqrt{\sum_i(X_i-\bar X)^2}\sqrt{\sum_i(Y_i-\bar Y)^2}}
\]

### Histogrammes

Avec $B=40$ classes :

\[
w=\frac{x_{\max}-x_{\min}}{B}
\]

\[
b_i=\operatorname{clamp}\left(
\left\lfloor\frac{x_i-x_{\min}}{w}\right\rfloor+1,1,B
\right)
\]

La hauteur d'une barre est le nombre de valeurs affectées à sa classe.

## Sources de référence

- `src/Model.cpp` : moments, reconstruction, KL et diagnostics ;
- `src/Model.hpp` : contrat des structures de métriques ;
- `scripts/tools/show-graph.lua` : filtrage des validations, statistiques,
  corrélations, histogrammes et rendu des rapports ;
- `src/Visualizer.cpp` et `src/HtopDisplay.hpp` : colonnes du CSV.
