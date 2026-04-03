# Jaguar Re-Identification — Kaggle

Competition : https://www.kaggle.com/competitions/jaguar-re-id

**Tâche** : étant donné deux photos de jaguars, prédire une similarité entre 0 et 1 indiquant s'il s'agit du même individu.  
**Dataset** : 1895 images train (31 individus), 371 images test, 137 270 paires à scorer.

---

## Architecture du projet

```
kaggle_jaguar_classification/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploration — ne touche à rien
│   ├── 02_data_cleaning.ipynb        # Nettoyage et préparation des données
│   ├── template_model.ipynb          # Template à dupliquer pour chaque modèle
│   └── XX_modelname_[baseline|improved_vN].ipynb
│
├── src/
│   ├── config.py                     # Constantes partagées (chemins, IMG_SIZE, SEED...)
│   └── data.py                       # Chargement des données + augmentation + pipeline tf.data
│
├── data/
│   ├── train/train/                  # Images brutes (originales)
│   ├── test/test/
│   ├── cleaned/train/                # Images nettoyées 224×224 RGB (générées par 02)
│   ├── cleaned/test/
│   ├── cleaned/train_split.csv       # Split 80% train
│   ├── cleaned/val_split.csv         # Split 20% val
│   └── cleaned/label_mapping.csv     # Nom jaguar ↔ entier (0–30)
│
├── outputs/                          # Graphiques générés par les notebooks
├── submissions/                      # Fichiers CSV à soumettre sur Kaggle
├── tensorboards/                     # Logs TensorBoard par modèle
└── requirements.txt
```

---

## Installation

```bash
python -m venv env
source env/bin/activate        # Windows : env\Scripts\activate
pip install -r requirements.txt
```

---

## Workflow

### 1. Exploration (01_eda.ipynb)
Lecture seule. On observe la distribution des classes, les dimensions des images, les outliers.  
**Ne rien modifier, ne rien sauvegarder.**

### 2. Nettoyage (02_data_cleaning.ipynb)
Produit les données prêtes à l'emploi dans `data/cleaned/` :
- Conversion RGBA → RGB (fond blanc)
- Resize 224×224 (Lanczos)
- Encodage des labels (LabelEncoder)
- Split stratifié 80/20

**À exécuter une seule fois.** Si on retouche ce notebook, on réexécute tout.

### 3. Modèles (notebooks XX_...)
Dupliquer `template_model.ipynb`, le renommer, et remplir les deux seules variables de config :

```python
MODEL_NAME   = 'linear_baseline'   # nom unique → utilisé pour logs, outputs, submission
AUGMENTATION = 'none'              # 'none', 'light', 'moderate', 'heavy'
```

Le reste du notebook est générique : il se branche automatiquement sur les données nettoyées via `src/data.py`.

### 4. Visualiser les runs TensorBoard

```bash
tensorboard --logdir=tensorboards/
```

Chaque run est sous `tensorboards/MODEL_NAME/run_name/`.

---

## Convention de nommage des notebooks

```
03_linear_baseline.ipynb
04_linear_improved_v1.ipynb
05_mlp_baseline.ipynb
06_cnn_baseline.ipynb
07_cnn_improved_v1.ipynb
...
```

On n'ajoute un `_improved_vN` que si le gain sur la val accuracy est significatif.  
Si le modèle est un dead-end, on le note dans la section Verdict du notebook et on passe au suivant.

---

## src/data.py — Fonctions disponibles

```python
from src.data import load_data, load_test_pairs, make_dataset

X_train, y_train        = load_data('train')   # images float32 + labels int
X_val,   y_val          = load_data('val')
X_test,  test_filenames = load_data('test')    # images + noms de fichiers
test_pairs_df           = load_test_pairs()    # DataFrame query/gallery

train_ds = make_dataset(X_train, y_train, training=True, augmentation='moderate')
val_ds   = make_dataset(X_val, y_val)          # pas d'augmentation sur la val
```

### Niveaux d'augmentation

| Niveau | Transformations |
|--------|----------------|
| `none` | Aucune (images brutes) |
| `light` | Flip horizontal |
| `moderate` | Flip + brightness + contrast |
| `heavy` | Flip + rotation (±29°) + zoom + brightness + contrast |

L'augmentation est **on-the-fly** : à chaque epoch, les transformations sont tirées aléatoirement, donc le modèle ne voit jamais exactement les mêmes images deux fois. C'est crucial sur un petit dataset comme celui-ci (1895 images).  
Le `tf.random.set_seed(SEED)` en début de notebook garantit la **reproductibilité** : relancer le notebook depuis le début donne toujours les mêmes résultats.

---

## Structure d'un notebook modèle

| Section | Contenu |
|---------|---------|
| **0. Header** | Architecture, objectif, diff vs version précédente |
| **1. Setup** | Imports, config (MODEL_NAME, AUGMENTATION), chargement données |
| **2. Architecture** | `build_model()` à définir |
| **3. HP Sweep** | LR sweep automatique sur `[1e-2, 1e-3, 1e-4]`, tableau comparatif |
| **4. Evaluation** | Courbes loss/acc, métriques, matrice de confusion |
| **5. Submission** | Retrain full dataset, embeddings + cosine similarity, export CSV |
| **6. Verdict** | Tableau résultats, analyse, verdict, score Kaggle public/privé |

### Choix fixes dans le template (non configurables)

- **Loss** : `sparse_categorical_crossentropy` — classification multi-classe avec labels entiers, pas d'alternative.
- **Metrics** : `accuracy` — métrique directe pour comparer les modèles entre eux.
- **Epochs** : `100` maximum — l'`EarlyStopping` arrête automatiquement quand la val loss stagne. Modifier le max n'aurait aucun effet en pratique.
- **Callbacks** : `EarlyStopping + ReduceLROnPlateau + TensorBoard` — combo standard, chaque callback a un rôle précis :
  - `EarlyStopping` : évite de perdre du temps et l'overfitting
  - `ReduceLROnPlateau` : affine la convergence quand on se rapproche du minimum
  - `TensorBoard` : traçabilité automatique, logs dans `tensorboards/`

### Ajouter un sweep sur d'autres hyperparamètres (notebooks improved)

Le sweep de base couvre le learning rate. Pour un improved, si on veut sweeper le dropout par exemple :

1. Modifier `build_model()` pour accepter le paramètre : `def build_model(dropout=0.4)`
2. Ajouter une boucle dans la section sweep :

```python
BEST_LR = ...  # déterminé par le sweep LR
for dr in [0.2, 0.4, 0.6]:
    m = build_model(dropout=dr)
    m.compile(optimizer=optimizers.Adam(BEST_LR), ...)
    results[f'dropout_{dr}'] = train_and_eval(...)
```

---

## Ce que produit chaque notebook

| Fichier | Destination |
|---------|-------------|
| `submission_MODEL_NAME.csv` | `submissions/` → à soumettre sur Kaggle |
| Courbes loss/acc, confusion, similarity | `outputs/` |
| Logs TensorBoard | `tensorboards/MODEL_NAME/` |

---

## Modèles explorés

| Notebook | Modèle | Val Acc | Score Kaggle (public) | Verdict |
|----------|--------|---------|-----------------------|---------|
| 03_linear_baseline | Linear | 72.8% | — | Solide lower-bound |
| 04_linear_improved_v1 | Linear + L2 + aug | 31.1% | — | Régression — à investiguer |
| 05_mlp_baseline | MLP | ~10% | — | Dead-end sans features |
| 06_cnn_baseline | CNN (tanh) | 9.5% | — | Cassé — tanh saturation |
| 07_resnet34_baseline | ResNet-34 | 81.0% | — | Bon |
| 08_resnet34_improved_v1 | ResNet-34 + BN + AdamW | 86.0% | — | Meilleur modèle actuel |
