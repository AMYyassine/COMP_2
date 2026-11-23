# IFT6390 — Competition 2 / Reproduction V10

Auteur : Assadiki Mohamed Yassine (20286183)
Équipe : `ift6390_mohamedyassine_assadiki`
Meilleur score Kaggle **public communiqué** : MAE 0.02808.

## Fichiers à placer dans le dossier data/
- `train_images.npy` : (2000,64,64), float32
- `train_labels.csv` : colonnes `id`, `log10_mass`
- `test_images.npy` : (1036,64,64), float32
- `sample_submission.csv` : facultatif

Ces données ne sont **pas incluses** dans le ZIP. Aucun fichier externe de lentilles ni poids préentraînés externes ne sont utilisés.

## Installation
Créer un environnement virtuel avec une version de Python compatible avec vos versions de PyTorch et SciPy. Sur Windows :

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Si PyTorch avec CUDA est déjà installé, conserver cette installation fonctionnelle ; consulter les instructions officielles PyTorch en cas de réinstallation GPU.

## Exécution / 2 modes

Ouvrir `notebook_final_V10.ipynb` dans VS Code ou Jupyter et **exécuter de haut en bas**.

1. **Reproduction fidèle avec checkpoints :** déposer, sous `results/`, `denoiser_full_v10.pt`, `boosting_v10_unet_geometry.joblib` et `extra_trees_v7.joblib` issus du projet original. Laisser `USE_SAVED_MODELS = True`. Le notebook produit `submissions/submission_v10_reproduite.csv` et compare avec `submissions/submission_v10_unet_geometry.csv` si présent (tolérance 1e-7). 
2. **Réentraînement depuis zéro :** si les fichiers checkpoints sont absents, le notebook les régénère sur les 2000 images autorisées (U-Net 65 époques, 5 augmentations/cible, Boosting70, ExtraTrees42), sauvegarder les modèles, puis créer le CSV. Les résultats peuvent différer légèrement selon versions et matériel GPU. Pour forcer le réentraînement : `USE_SAVED_MODELS = False`.

`RUN_CV = False` par défaut. Passer à `True` pour recalculer les cinq plis ; placer les checkpoints `results/denoiser_independent_fold1.pt` à `fold5.pt`, ou activer `RETRAIN_CV_DENOISERS_IF_MISSING=True` (calcul long). **Ce sont des débruiteurs indépendants des plis externes** ; ne pas utiliser le checkpoint final pour ces scores.

## Pipeline
- Bruit de train simulé `sigma=0.133942` et translations aléatoires `[-10,+10]` pixels.
- Extraction 42 caractéristiques radiales/géométriques des images bruitées + 14 sur image lissée + 14 sur image reconstruite par U-Net depuis zéro.
- `HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.05, l2_regularization=1.0)` sur 70 variables.
- `ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, max_features=0.85, random_state=42)` sur 42 variables.
- Combinaison : `0.70 × Boosting70 + 0.30 × ExtraTrees42`.
- Sortie : `submissions/submission_v10_reproduite.csv`, colonnes `id,log10_mass`, ids 0..1035.


