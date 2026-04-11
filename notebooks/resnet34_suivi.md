# Suivi des expérimentations — ResNet-34

> **But** : documenter l'évolution des notebooks ResNet-34, les résultats obtenus et les choix faits pour la soutenance.  
> **Mise à jour** : à chaque nouveau notebook.

---

## Vue d'ensemble

| Notebook | Modèle | Val Acc | Train Acc | Overfitting | Kaggle (pub / priv) |
|----------|--------|---------|-----------|-------------|---------------------|
| 06_0 | Baseline (tanh, Adam) | 81.00 % | 90.37 % | Modéré (gap 9 pts) | 0.292 / 0.277 |
| 06_1 | BN + ReLU + AdamW | 86.02 % | 100.00 % | Sévère (gap 14 pts) | 0.530 / 0.485 |
| 06_2 | + LR sweep + wd=5e-4 + bs=64 | 88.65 % | 100.00 % | Sévère (gap 11 pts) | 0.566 / 0.542 |
| 06_3 | + Augmentation moderate (pipeline) | 88.13 % | 100.00 % | Sévère (gap 12 pts) | **0.278 / 0.253** |

**Contexte dataset** : 1516 images d'entraînement, 379 de validation, 31 classes (jaguars).  
**Ratio paramètres/images** : 21M params / 1516 images = **~14 000:1** (pour comparaison, ImageNet = ~17:1). Ce ratio extrême rend l'overfitting quasi-inévitable sans régularisation agressive.

---

## 06_0 — ResNet-34 Baseline

**Notebook** : `06_0_resnet34_baseline.ipynb`

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Activation | `tanh` (pas de BatchNorm) |
| Optimizer | `Adam(lr=1e-3)` |
| Batch size | 32 |
| Epochs | 50 (EarlyStopping patience=10) |
| Augmentation | ✗ |
| LR sweep | ✗ |
| Weight decay | ✗ |

### Résultats
| Métrique | Train | Val |
|----------|-------|-----|
| Accuracy | 90.37 % | 81.00 % |
| Loss | 0.3914 | 0.9792 |
| Best epoch | 50 / 50 | |
| Embedding dim | 512 | |
| Kaggle public | 0.292 | |
| Kaggle private | 0.277 | |

### Analyse

- L'accuracy de validation à **81 %** est un bon point de départ pour un ResNet-34 entraîné from scratch sur un dataset aussi petit.
- **Best epoch = 50 / 50** : le modèle n'a pas convergé. L'EarlyStopping (patience=10) ne s'est jamais déclenché → la val loss continuait de s'améliorer. L'entraînement a été coupé trop tôt.
- **Gap train/val = 9 points** (90 % vs 81 %) : overfitting modéré. Le modèle commence à mémoriser les images d'entraînement mais pas encore de façon catastrophique.
- **Loss val = 0.9792** est relativement élevée, signe que le modèle n'est pas très confiant dans ses prédictions.
- **`tanh` est inadapté aux ResNets profonds** : cette activation sature pour les valeurs extrêmes (gradient → 0), ce qui provoque le vanishing gradient dans un réseau de 34 couches. Les activations traversent 2 convolutions par bloc × 16 blocs = 32 activations tanh → le gradient s'atténue drastiquement. C'est la raison principale pour laquelle le réseau original ResNet (He et al., 2015) utilise ReLU + BatchNorm.
- **Pas de régularisation** : ni weight decay, ni dropout, ni augmentation. Avec un ratio paramètres/images de 14 000:1, le modèle a toute latitude pour mémoriser.

### Verdict
**Point de départ solide.** Les problèmes identifiés forment la feuille de route pour 06_1 :
1. Activation inadaptée (tanh) → passer à BN + ReLU
2. Pas de régularisation → ajouter Dropout + weight decay
3. Entraînement trop court → augmenter le nombre d'epochs

---

## 06_1 — ResNet-34 Improved (BN + AdamW)

**Notebook** : `06_1_resnet34_improved.ipynb`

### Pourquoi ces changements ?

Chaque modification cible un problème identifié dans 06_0 :

| Changement | Problème visé | Raisonnement |
|-----------|--------------|--------------|
| `tanh` → `BatchNorm → ReLU` | Vanishing gradient | `tanh` sature (gradient → 0) dans les 34 couches. `ReLU` a un gradient constant de 1 pour x > 0. `BatchNorm` normalise les activations avant chaque ReLU → le gradient se propage efficacement. |
| Shortcut : `Conv1x1` → `Conv1x1 + BN` | Shortcut non normalisé | Sans BN sur le shortcut, l'addition résiduelle mélange des échelles différentes. BN harmonise les deux branches. |
| Ajout de `Dropout(0.4)` | Overfitting (ratio 14 000:1) | Le modèle a 21M de params pour 1516 images : il peut mémoriser chaque image individuellement. Dropout éteint aléatoirement 40 % des neurones → force le modèle à distribuer l'information de façon redondante. |
| `Adam` → `AdamW(lr=1e-3, wd=1e-4)` | Régularisation L2 couplée au LR | Avec Adam classique, la régularisation L2 est multipliée par le learning rate. Quand `ReduceLROnPlateau` réduit le LR, le poids de la régularisation diminue aussi → elle perd son effet en fin d'entraînement. `AdamW` découple les deux : le weight decay reste constant indépendamment du LR. |
| Epochs 50 → 100 (patience=20) | Convergence incomplète | Best epoch = 50/50 dans 06_0 → le modèle n'avait pas fini de converger. 100 epochs + `ReduceLROnPlateau` (factor=0.5, patience=7) permet d'affiner la convergence. `EarlyStopping` (patience=20) protège du surapprentissage prolongé. |

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Activation | `BatchNorm → ReLU` |
| Optimizer | `AdamW(lr=1e-3, wd=1e-4)` |
| Batch size | 32 |
| Epochs max | 100 (EarlyStopping patience=20) |
| Augmentation | ✗ |
| LR sweep | ✗ |

### Résultats
| Métrique | Train | Val |
|----------|-------|-----|
| Accuracy | 100.00 % | 86.02 % |
| Loss | 0.0003 | 0.7877 |
| Best epoch | 38 / 100 | |
| Embedding dim | 512 | |
| Kaggle public | 0.530 | |
| Kaggle private | 0.485 | |

### Analyse

- **Val accuracy : 81 % → 86 %** (+5 points). L'amélioration vient principalement de `BN + ReLU` : le gradient circule mieux, le modèle apprend des features plus discriminantes.
- **Convergence 2× plus rapide** : best epoch = 38 (vs 50 qui n'avait pas convergé dans 06_0). BN accélère l'entraînement en normalisant les activations → chaque couche reçoit des inputs bien calibrés, ce qui permet un learning rate plus agressif.
- **Train accuracy = 100 %** : le modèle mémorise parfaitement les 1516 images d'entraînement. La loss train = 0.0003 (quasi-zéro) confirme que le modèle est extrêmement confiant sur les données vues.
- **Gap train/val = 14 points** (100 % vs 86 %) : l'overfitting s'est aggravé par rapport à 06_0 (9 pts). Paradoxalement, améliorer l'architecture a augmenté l'overfitting : le modèle est désormais assez puissant pour tout mémoriser. `Dropout(0.4)` et `wd=1e-4` ne suffisent pas.
- **Val loss = 0.7877** (vs 0.9792 dans 06_0) : le modèle généralise un peu mieux en moyenne, mais l'écart reste important. La loss val est ~2600× la loss train (0.7877 / 0.0003) : mémorisation totale.

### Verdict
**L'architecture corrigée (BN + ReLU) est clairement bénéfique (+5 % val).** Mais l'overfitting empire. Les régularisations actuelles (Dropout 0.4 + wd 1e-4) sont insuffisantes face au ratio paramètres/images extrême. Pour 06_2, deux axes :
1. Renforcer la régularisation (weight decay plus fort)
2. Optimiser les hyperparamètres (LR sweep, batch size)

---

## 06_2 — ResNet-34 Improved v2 (LR sweep + régularisation renforcée)

**Notebook** : `06_2_resnet34_improved.ipynb`

### Pourquoi ces changements ?

| Changement | Problème visé | Raisonnement |
|-----------|--------------|--------------|
| Weight decay 1e-4 → **5e-4** (×5) | Overfitting sévère (train=100 %) | `wd=1e-4` n'a pas empêché la mémorisation totale. On multiplie par 5 pour pénaliser plus fortement les poids élevés, forçant le modèle vers des solutions plus simples et mieux généralisables. |
| Batch size 32 → **64** | Gradient bruité | Avec bs=32, chaque batch = 2 % du dataset (32/1516) → le gradient estimé est très bruité, ce qui peut nuire à la convergence et la généralisation. Avec bs=64 = 4 % du dataset → gradient plus stable, moins de variance inter-batch. Bonus : entraînement plus rapide (moins de steps par epoch). |
| **LR sweep** `[1e-3, 5e-4, 1e-4]` | LR choisi arbitrairement | Dans 06_1, `lr=1e-3` était un choix par défaut. Le sweep teste systématiquement 3 valeurs pour trouver l'optimum. C'est une bonne pratique : le LR est l'hyperparamètre le plus impactant sur la convergence. |
| `AUGMENTATION='none'` (volontaire) | Isoler l'effet des changements | En gardant l'augmentation à `none`, on mesure uniquement l'impact du weight decay renforcé + batch size + LR sweep, sans confondre avec l'effet de l'augmentation. Principe : **changer une variable à la fois**. |

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Optimizer | `AdamW(wd=5e-4)` |
| Batch size | 64 |
| LR sweep | `[1e-3, 5e-4, 1e-4]` |
| Best LR | `1e-3` |
| Augmentation | `none` |

### LR Sweep
| LR | Val Accuracy | Best Epoch | Observation |
|----|-------------|------------|-------------|
| 1e-3 | **89.18 %** | 61 | Meilleur compromis vitesse/performance |
| 5e-4 | 86.81 % | 38 | Converge plus vite mais plafonne plus bas |
| 1e-4 | 0.79 % | 1 | Ne converge pas du tout (LR trop faible) |

**Interprétation du LR sweep** :
- `lr=1e-3` est le sweet spot : assez fort pour sortir des minima locaux peu profonds, mais pas trop fort pour diverger. Il met 61 epochs pour converger (vs 38 pour 06_1) car le weight decay plus fort ralentit l'apprentissage — c'est normal et souhaitable.
- `lr=5e-4` converge en 38 epochs mais plafonne 2.4 points plus bas (86.81 % vs 89.18 %). Le LR n'est pas assez fort pour explorer suffisamment l'espace des poids.
- `lr=1e-4` est un échec total (0.79 %) : le modèle reste coincé dans la configuration aléatoire initiale. Les updates de poids sont trop faibles pour que le réseau apprenne quoi que ce soit en 100 epochs. `ReduceLROnPlateau` ne peut pas compenser un LR déjà trop faible au départ.

### Résultats (best LR = 1e-3)
| Métrique | Train | Val |
|----------|-------|-----|
| Accuracy | 100.00 % | 88.65 % |
| Loss | 0.0001 | 0.7121 |
| Best epoch | 61 / 100 | |
| Kaggle public | 0.566 | |
| Kaggle private | 0.542 | |

### Analyse

- **Val accuracy : 86 % → 88.65 %** (+2.63 points). Progression plus modeste que 06_0→06_1, mais significative. Elle vient de la combinaison weight decay ×5 + batch size ×2.
- **Weight decay 5e-4** : la val loss passe de 0.7877 à 0.7121 (-10 %). Le modèle produit des poids plus petits → prédictions moins extrêmes → meilleure généralisation. Mais ça ne suffit pas à empêcher train=100 %.
- **Batch size 64** : gradient plus stable → le modèle ne zigzague pas autant entre les minibatchs. Sur un dataset très petit, c'est important car les batchs de 32 peuvent être non-représentatifs (certaines classes absentes d'un batch donné).
- **Train = 100 % persiste** : malgré wd ×5, le modèle mémorise toujours tout. La gap train/val = 11 points (vs 14 dans 06_1) → léger progrès, mais pas suffisant.
- **Constat clé** : la régularisation par les poids (weight decay + dropout) a ses limites. Avec 21M de paramètres et seulement 1516 images, le modèle a trop de capacité. La prochaine étape logique : **augmenter artificiellement la taille du dataset** via la data augmentation.

### Verdict
**Meilleure val accuracy jusqu'ici (88.65 %).** Le LR sweep est une bonne pratique qui sera réutilisée dans tous les notebooks suivants. Mais l'overfitting reste sévère (train=100 %). La régularisation seule (wd + dropout) ne peut pas compenser un ratio paramètres/images de 14 000:1. La data augmentation est la prochaine étape naturelle.

---

## 06_3 — ResNet-34 Improved v3 (augmentation moderate via pipeline)

**Notebook** : `06_3_resnet34_improved.ipynb`

### Pourquoi ces changements ?

| Changement | Problème visé | Raisonnement |
|-----------|--------------|--------------|
| `AUGMENTATION = 'moderate'` | Train=100 %, régularisation insuffisante | La régularisation des poids (wd, dropout) a atteint ses limites. La data augmentation attaque le problème différemment : au lieu de contraindre les poids, on enrichit les données. Chaque image est légèrement différente à chaque epoch (flip, luminosité, contraste) → le modèle ne peut plus mémoriser les pixels exacts. |
| Pipeline `make_dataset()` au lieu de layers dans le modèle | Bug Keras : propagation `training` flag | Tentative initiale : insérer les layers `RandomFlip`, `RandomBrightness`, etc. directement dans le modèle Keras fonctionnel. **Bug identifié** : `RandomBrightness(0.2)` sans `value_range=(0.0, 1.0)` sur des images normalisées [0,1] corrompt les pixels (le layer croit que les pixels sont dans [0, 255] et ajuste la luminosité en conséquence → valeurs aberrantes). Utiliser `make_dataset()` de `src/data.py` résout le problème : `get_augmentation()` spécifie correctement `value_range=(0.0, 1.0)`, et l'augmentation est appelée avec `aug(x, training=True)` explicitement dans le pipeline `tf.data`. |
| Modèle simplifié (pas de Random* dans le graphe) | Séparation des responsabilités | Le modèle est un pur ResNet-34 sans logique d'augmentation. L'augmentation est gérée par le pipeline de données → plus lisible, plus maintenable, plus facile à tester. |

### Comment fonctionne l'augmentation `moderate`

À chaque batch pendant `fit()`, les 3 transformations sont appliquées aléatoirement :
1. **RandomFlip('horizontal')** : miroir horizontal avec probabilité 50 %. Justification : les jaguars sont symétriques, un flip ne change pas l'identité.
2. **RandomBrightness(0.2, value_range=(0,1))** : luminosité ±20 %. Justification : les photos sont prises dans des conditions d'éclairage variées (jour/nuit, ombre/soleil). Le modèle doit être robuste à ces variations.
3. **RandomContrast(0.2)** : contraste ±20 %. Justification : les caméras-pièges produisent des images avec des contrastes très variables.

Pendant `evaluate()` et `predict()`, ces transformations sont désactivées → les métriques et prédictions sont sur les images originales.

### Configuration
| Paramètre | Valeur |
|-----------|--------|
| Optimizer | `AdamW(lr=best, wd=5e-4)` |
| Batch size | 64 |
| LR sweep | `[1e-3, 5e-4, 1e-4]` |
| Augmentation | `moderate` (flip + brightness + contrast) |
| Pipeline | `make_dataset()` de `src/data.py` |

### LR Sweep
| LR | Val Accuracy | Best Epoch | Observation |
|----|-------------|------------|-------------|
| 1e-3 | 86.28 % | 71 | Plus lent et moins bon qu'en 06_2 (89.18 %) |
| 5e-4 | **86.81 %** | 39 | Meilleur LR avec augmentation |
| 1e-4 | 8.97 % | 1 | Toujours pas convergé |

**Changement notable** : le best LR passe de `1e-3` (06_2, sans augmentation) à `5e-4` (06_3, avec augmentation). L'augmentation introduit du bruit dans les données → un LR plus petit stabilise l'apprentissage. Avec `lr=1e-3`, les updates de poids sont trop agressives face aux variations ajoutées par l'augmentation → le modèle oscille davantage et finit plus bas (86.28 % vs 86.81 %).

### Résultats (best LR = 5e-4)
| Métrique | Train | Val |
|----------|-------|-----|
| Accuracy | 100.00 % | 88.13 % |
| Loss | 0.0001 | 0.7013 |
| Best epoch | 39 / 100 | |
| Embedding dim | 512 | |
| Kaggle public | **0.278** | |
| Kaggle private | **0.253** | |

### Analyse

**La val accuracy est stable, mais le score Kaggle s'effondre.** C'est le résultat le plus surprenant et le plus instructif de cette série.

- **Val accuracy = 88.13 %** : quasi-identique à 06_2 (88.65 %, -0.52 pts). En termes de classification, l'augmentation ne change rien. Le modèle classe toujours aussi bien les jaguars dans les 31 catégories.
- **Score Kaggle = 0.278 / 0.253** : effondrement par rapport à 06_2 (0.566 / 0.542, soit **-50 %**). Le score Kaggle est pire que la baseline 06_0 (0.292 / 0.277). C'est un résultat contre-intuitif : le modèle classe mieux mais ses embeddings sont inutilisables pour la similarité.
- **Train = 100 % persiste** : l'augmentation `moderate` est trop faible pour empêcher la mémorisation.
- **Best LR shift 1e-3 → 5e-4** : l'augmentation change le landscape de la loss. Leçon : **toujours refaire un LR sweep quand on modifie l'augmentation.**

### Interprétation — pourquoi le score Kaggle s'effondre

C'est la leçon la plus importante de toute la série ResNet-34 : **classification accuracy ≠ qualité des embeddings.**

Le score Kaggle mesure la **similarité cosine** entre embeddings de paires d'images. La val accuracy mesure la **classification** en 31 classes. Ces deux métriques évaluent des choses différentes :

1. **Classification** : le modèle doit séparer 31 classes. Il suffit que les clusters soient séparés, peu importe leur forme ou leur compacité.
2. **Similarité (Re-ID)** : les embeddings du même jaguar doivent être proches, ceux de jaguars différents doivent être éloignés. Les clusters doivent être **compacts** et **bien séparés** dans l'espace des embeddings.

**Pourquoi l'augmentation dégrade les embeddings :**
- L'augmentation introduit de la variabilité dans les images d'entraînement → le modèle apprend des features plus "floues", moins discriminantes au niveau pixel.
- Le modèle final (retrained sur `all_ds` avec augmentation) est entraîné seulement **39 epochs** (BEST_EPOCH du sweep) — contre 61 pour 06_2. Moins d'epochs + données bruitées = embeddings de moins bonne qualité.
- Les transformations de luminosité/contraste peuvent modifier des caractéristiques visuelles importantes pour distinguer les jaguars (contrastes des rosettes, texture du pelage). Le modèle perd en finesse de discrimination.
- L'augmentation est appliquée aussi pendant le **retrain final** (`all_ds` utilise `augmentation=AUGMENTATION`). Les embeddings extraits sont basés sur un modèle qui n'a jamais vu les images "propres" dans leur forme originale pendant le retrain.

**Leçon fondamentale** : pour une tâche de Re-ID, il ne faut pas optimiser que la classification. Les embeddings sont ce qui compte pour le score final. Une augmentation qui aide la classification peut détruire la qualité des embeddings. Il faudrait :
- Utiliser une **loss métrique** (triplet, ArcFace) qui optimise directement la structure de l'espace des embeddings
- Ou au minimum, faire le retrain final **sans augmentation** pour que les embeddings soient extraits d'un modèle entraîné sur les images propres

---

## Récapitulatif des leçons apprises

| Changement | Impact sur val | Leçon retenue |
|-----------|---------------|---------------|
| `tanh` → `BN + ReLU` | **+5 pts** (81→86 %) | L'architecture est le facteur le plus impactant. BN + ReLU est le standard pour les ResNets — ne pas réinventer la roue. |
| `Adam` → `AdamW(wd=1e-4)` | inclus dans +5 pts | AdamW découple weight decay et LR → régularisation stable même quand le LR diminue. |
| `wd` 1e-4 → 5e-4 + `bs` 32 → 64 | **+2.6 pts** (86→88.65 %) | La régularisation par les poids améliore la généralisation, mais ne peut pas compenser un ratio params/images extrême (14 000:1). |
| LR sweep `[1e-3, 5e-4, 1e-4]` | confirme `lr=1e-3` optimal | Le LR est l'hyperparamètre le plus sensible. `1e-4` ne converge pas du tout. Toujours faire un sweep avant de fixer le LR. |
| Augmentation `moderate` (pipeline) | **-0.5 pts** val, **Kaggle -50 %** | L'augmentation dégrade les embeddings même si la classification est stable. Val accuracy ≠ qualité des embeddings. Le retrain final avec augmentation nuit à la similarité cosine. |

### Évolution du gap train/val

| Notebook | Train Acc | Val Acc | Gap | Loss ratio (val/train) | Kaggle (pub) |
|----------|-----------|---------|-----|----------------------|-------------|
| 06_0 | 90.37 % | 81.00 % | 9.4 pts | 2.5× | 0.292 |
| 06_1 | 100.00 % | 86.02 % | 14.0 pts | 2 626× | 0.530 |
| 06_2 | 100.00 % | 88.65 % | 11.4 pts | 7 121× | **0.566** |
| 06_3 | 100.00 % | 88.13 % | 11.9 pts | 7 013× | **0.278** ↘ |

Le loss ratio val/train explose à partir de 06_1. Mais le fait le plus marquant est le **décrochage Kaggle de 06_3** : la val accuracy est quasi-stable (88.13 % vs 88.65 %) mais le score Kaggle chute de moitié (0.566 → 0.278). Cela démontre que **la val accuracy n'est pas un bon proxy du score Kaggle** pour ce problème de Re-ID. L'augmentation dégrade la qualité des embeddings même quand la classification reste bonne.

---

## Pistes pour la suite

### 1. Augmentation `heavy`
**Quoi** : ajouter rotation (±8°), zoom (±10 %) en plus de flip + brightness + contrast.  
**Pourquoi** : l'augmentation `moderate` ne touche qu'à l'apparence (luminosité, contraste). L'augmentation `heavy` modifie aussi la géométrie (rotation, zoom) → le modèle doit apprendre des features invariantes à la position et à l'échelle.  
**Risque** : trop d'augmentation peut rendre l'apprentissage instable si les images transformées deviennent trop différentes des originales. À surveiller : si la val accuracy baisse → réduire l'intensité des transformations.

### 2. Transfer learning (pré-entraîné ImageNet)
**Quoi** : remplacer le ResNet-34 from scratch par un ResNet-34 (ou ResNet-50) pré-entraîné sur ImageNet. Fine-tuner les dernières couches + le head de classification.  
**Pourquoi** : les premières couches d'un CNN apprennent des features universelles (contours, textures, motifs répétitifs). Sur un dataset de 1516 images, apprendre ces features from scratch est un gaspillage — un modèle pré-entraîné sur 1.2M d'images les fournit gratuitement. Le fine-tuning n'a besoin d'adapter que les couches de haut niveau (formes spécifiques aux jaguars, patterns de rosettes).  
**Impact attendu** : gains importants (+5-10 % val) car le modèle part de features déjà utiles au lieu de poids aléatoires.  
**Implémentation** : `keras.applications.ResNet50(weights='imagenet', include_top=False)` + `GlobalAveragePooling2D` + `Dense(NUM_CLASSES)`. Freeze des premières couches, unfreeze progressif.

### 3. Loss métrique (triplet / contrastive / ArcFace)
**Quoi** : remplacer `sparse_categorical_crossentropy` par une loss qui optimise directement la similarité entre embeddings.  
**Pourquoi** : la cross-entropy entraîne un classifieur à 31 classes, mais le problème final est de comparer des **paires d'images** (même jaguar ou pas). La cross-entropy force le modèle à séparer 31 classes dans l'espace des embeddings, mais elle n'optimise pas directement la distance entre paires → les embeddings peuvent être bien séparés par classe mais mal calibrés pour la similarité cosine.  
**Triplet loss** : prend un anchor, un positif (même classe) et un négatif (classe différente). Force la distance anchor-positif < anchor-négatif. Optimise directement ce qu'on mesure à la soumission Kaggle.  
**ArcFace / CosFace** : projettent les embeddings sur une hypersphère → la similarité cosine devient une mesure de distance angulaire. Souvent supérieur à la triplet loss car plus stable à entraîner.  
**Impact attendu** : amélioration significative du score Kaggle (la métrique de soumission est basée sur la similarité, pas sur la classification).

### 4. Label smoothing
**Quoi** : remplacer les labels one-hot (0 ou 1) par des distributions adoucies (0.9 pour la vraie classe, 0.1/30 ≈ 0.003 pour les autres).  
**Pourquoi** : les labels one-hot encouragent le modèle à être confiant à 100 % → il pousse les logits vers l'infini → overfitting des poids. Le label smoothing met un plafond implicite à la confiance → les poids restent raisonnables, les embeddings sont mieux calibrés.  
**Implémentation** : `loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1)` (attention : nécessite `CategoricalCrossentropy` avec labels one-hot, pas sparse).  
**Impact attendu** : modeste (+0.5-1 % val) mais simple à implémenter et compatible avec toutes les autres optimisations.

### 5. Dropout 0.4 → 0.5
**Quoi** : augmenter le taux de dropout.  
**Pourquoi** : si l'overfitting persiste après augmentation, c'est un levier simple à actionner. Éteindre 50 % des neurones au lieu de 40 % force plus de redondance.  
**Risque** : un dropout trop élevé ralentit la convergence et peut empêcher le modèle d'apprendre. Si la val accuracy baisse avec Dropout(0.5) → revenir à 0.4.  
**Impact attendu** : faible, surtout si l'augmentation résout déjà l'overfitting.

### 6. Architectures alternatives
**EfficientNet** : architecture optimisée par NAS (Neural Architecture Search). Plus efficace en params que ResNet → meilleur ratio performance/taille. EfficientNet-B0 = 5.3M params (vs 21M pour ResNet-34) pour des performances comparables.  
**Vision Transformer (ViT)** : attention maps captent les relations spatiales globales → potentiellement meilleur pour les détails fins comme les motifs de rosettes des jaguars. Mais nécessite beaucoup de données → viable uniquement en transfer learning (ViT-B/16 pré-entraîné).

### Priorité recommandée
1. ~~**06_3** : valider l'augmentation `moderate`~~ → fait, impact quasi-nul
2. **Augmentation `heavy`** : tester si les transformations géométriques (rotation, zoom) font la différence
3. **Transfer learning** : c'est le levier avec le plus gros impact attendu sur un petit dataset — les gains sont immédiats
4. **Loss métrique** : aligner l'entraînement avec la métrique Kaggle (similarité, pas classification)
5. Le reste (label smoothing, dropout) = optimisations incrémentales

---

*Dernière mise à jour : 06_3 exécuté — val=88.13 %, augmentation moderate sans impact significatif.*
