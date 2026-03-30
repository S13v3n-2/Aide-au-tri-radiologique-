# Aide au tri radiologique — Deep Learning

Projet académique de classification de pathologies thoraciques par deep learning.
**Keras 3 / TensorFlow 2.15+ · MLflow · Streamlit**

---

## Présentation

Ce projet implémente un pipeline complet d'analyse d'images thoraciques :

- **Classification supervisée** — CNN scratch, DenseNet-121, EfficientNetV2-S, ViT
- **Détection d'anomalies** — Autoencodeur convolutionnel (ConvAE) et VAE
- **Modélisation multimodale** — Fusion tardive image + compte-rendu textuel (TF-IDF)
- **Suivi expérimental** — Tracking complet via MLflow
- **Interface interactive** — Application Streamlit de démonstration

---

## Structure du projet

```
Projet_Deep_Learning/
├── 01_eda_data_preparation.ipynb         # EDA + export numpy + pipeline tf.data
├── 02_supervised_classification.ipynb   # CNN scratch, DenseNet-121, EfficientNetV2-S, ViT
├── 03_anomaly_detection_ae_vae.ipynb    # ConvAE + ConvVAE Keras
├── 04_multimodal_image_text.ipynb       # Fusion tardive image + TF-IDF (NIH)
├── 05_evaluation_mlflow.ipynb           # Évaluation finale + traçabilité MLflow
├── streamlit_app.py                     # Démonstrateur Streamlit
├── data/                                # Données (ChestMNIST auto, NIH manuel)
├── models/                              # Checkpoints .keras sauvegardés
├── requirements.txt
└── README.md
```

---

## Résultats

| Modèle | ROC-AUC | PR-AUC | F1-macro |
|---|---|---|---|
| **EfficientNetV2-S** | **0.7684** | **0.158** | **0.168** |
| DenseNet-121 | 0.506 | — | — |
| CNN scratch | ~0.50 | — | — |
| ViT-Small | ~0.50 | — | — |
| ConvVAE (anomalie) | 0.52 | — | — |

> Seul EfficientNetV2-S apprend réellement sur ChestMNIST 64×64. Les modèles multimodaux
> affichent AUC 1.000 à cause d'une fuite de données (texte généré depuis les labels).

---

## Installation

```bash
pip install -r requirements.txt
```

## Données

**ChestMNIST** — téléchargement automatique via `medmnist` dans le notebook 01.

**NIH ChestX-ray14** — requis pour le notebook 04 (multimodal) :
- En local : télécharger depuis [nihcc.app.box.com](https://nihcc.app.box.com/v/ChestXray-NIHCC), placer `Data_Entry_2017.csv` dans `data/` et les images dans `data/nih_images/`
- Sur Colab : utiliser `drive/00_download_nih_kaggle_colab.ipynb` (télécharge ~45 Go depuis Kaggle vers Google Drive)

---

## Ordre d'exécution

```
01_eda_data_preparation.ipynb
    ↓  exporte X_*.npy, y_*.npy, config.json
02_supervised_classification.ipynb
    ↓  exporte models/best_supervised_model.keras
03_anomaly_detection_ae_vae.ipynb
    ↓  exporte models/ConvAE_best.keras + ConvVAE_best.keras
04_multimodal_image_text.ipynb       ← requiert données NIH
    ↓
05_evaluation_mlflow.ipynb           ← rapport final + traçabilité
```

## Lancer l'interface

```bash
streamlit run streamlit_app.py
```

## Suivi MLflow

```bash
mlflow ui --port 5000
```

Expériences créées :
- `chest_classification_supervised` — CNN scratch, DenseNet-121, EfficientNetV2-S, ViT
- `chest_anomaly_detection` — ConvAE, ConvVAE
- `chest_multimodal` — ImageOnly, TextOnly, LateFusion

---

## Points techniques

| Aspect | Implémentation |
|---|---|
| Format modèle | `.keras` (Keras 3 natif) |
| Augmentation | `keras.layers.Random*` intégré au graphe |
| Pipeline données | `tf.data.Dataset` + `.prefetch(AUTOTUNE)` |
| Perte multi-label | `tf.nn.weighted_cross_entropy_with_logits` avec pos_weight |
| Fine-tuning | 2 phases : backbone gelé (15 ep) → dégel partiel (45 ep) |
| VAE | `train_step` / `test_step` overridés dans `keras.Model` |
| ViT | Implémentation native Keras (`MultiHeadAttention`, `Embedding`) |
| TPU | `TPUStrategy` + `drop_remainder=True` + `bfloat16` + LR scaling |

## Reproductibilité

- `SEED = 42` propagé via `keras.utils.set_random_seed(SEED)`
- Splits officiels MedMNIST conservés (pas de re-split)
- Split patient-level pour NIH (anti-fuite inter-patient)
- Arrays numpy exportés dans `data/` pour réutilisation entre notebooks

## Configuration matérielle de référence

L'ensemble des modèles a été entraîné et validé sur la configuration suivante pour garantir la reproductibilité des temps de calcul :
* **GPU** : NVIDIA GeForce RTX 3070
* **VRAM** : 8 Go
* **Architecture** : Ampere

## Configuration matérielle

| | Minimum | Recommandé | Colab gratuit |
|---|---|---|---|
| RAM | 8 Go | 16 Go | 12 Go (GPU) / 335 Go (TPU) |
| GPU | CPU only | GPU 6 Go+ | T4 (GPU) / v2-8 (TPU) |
| Stockage | 5 Go (ChestMNIST) | 50 Go (NIH complet) | Google Drive |

---
## Analyse des résultats et Traçabilité

Le pilotage du projet s'appuie sur le score ROC-AUC Macro comme métrique de référence, cette dernière étant particulièrement robuste pour la classification multi-label en imagerie médicale. La traçabilité complète, incluant les courbes de précision-rappel et les rapports de classification par pathologie, est générée automatiquement dans le module d'évaluation finale