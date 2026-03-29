# Système d'aide au tri radiologique — Keras / TensorFlow

Projet Deep Learning — Classification de pathologies thoraciques, détection d'anomalies, modélisation multimodale.
**Framework : Keras 3 / TensorFlow 2.15+**

## Structure du projet

```
projet_radiology/
├── notebooks/
│   ├── 01_eda_data_preparation.ipynb       # EDA + export numpy + tf.data
│   ├── 02_supervised_classification.ipynb  # CNN scratch, DenseNet-121, EfficientNetV2-S, ViT
│   ├── 03_anomaly_detection_ae_vae.ipynb   # ConvAE + ConvVAE Keras
│   ├── 04_multimodal_image_text.ipynb      # Fusion tardive image + TF-IDF (NIH)
│   └── 05_evaluation_mlflow.ipynb          # Évaluation finale + traçabilité MLflow
├── app/
│   └── streamlit_app.py                    # Démonstrateur Streamlit (Keras)
├── data/                                   # Données (ChestMNIST auto, NIH manuel)
├── models/                                 # Checkpoints .keras sauvegardés
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Données

**ChestMNIST** — téléchargement automatique via `medmnist` dans le notebook 01.

**NIH ChestX-ray14** — requis pour le notebook 04 (multimodal) :
1. Télécharger depuis https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Placer `Data_Entry_2017.csv` dans `data/`
3. Placer les images dans `data/nih_images/`

## Ordre d'exécution

```
01_eda_data_preparation.ipynb
    ↓ (exporte X_*.npy, y_*.npy, config.json)
02_supervised_classification.ipynb
    ↓ (exporte models/best_supervised_model.keras)
03_anomaly_detection_ae_vae.ipynb
    ↓ (exporte models/ConvAE_best.keras + ConvVAE_best.keras)
04_multimodal_image_text.ipynb    [requiert données NIH]
    ↓
05_evaluation_mlflow.ipynb        [rapport final + traçabilité]
```

## Démarrage du démonstrateur

```bash
streamlit run streamlit_app.py
```

## Suivi MLflow

```bash
mlflow ui --port 5000
# Interface sur http://localhost:5000
```

Expériences MLflow créées :
- `chest_classification_supervised` — CNN scratch, DenseNet-121, EfficientNetV2-S, ViT
- `chest_anomaly_detection` — ConvAE, ConvVAE
- `chest_multimodal` — ImageOnly, TextOnly, LateFusion

## Points techniques clés

| Aspect | Choix Keras/TF |
|---|---|
| Format modèle | `.keras` (nouveau format natif Keras 3) |
| Augmentation | `keras.layers.Random*` intégré au graphe |
| Pipeline données | `tf.data.Dataset` avec `.prefetch(AUTOTUNE)` |
| Perte multi-label | `tf.nn.weighted_cross_entropy_with_logits` |
| VAE | `train_step`/`test_step` overridés dans `keras.Model` |
| Transfer learning | `keras.applications` (DenseNet121, EfficientNetV2*) |
| ViT | Implémentation native Keras (`MultiHeadAttention`, `Embedding`) |

## Reproductibilité

- `SEED = 42` propagé via `keras.utils.set_random_seed(SEED)`
- Splits officiels MedMNIST conservés (pas de re-split)
- Split patient-level pour NIH (anti-fuite inter-patient)
- Arrays numpy exportés dans `data/` pour réutilisation entre notebooks

## Configuration matérielle recommandée

| Composante | Minimum | Recommandé |
|---|---|---|
| RAM | 8 Go | 16 Go |
| GPU | CPU only (lent) | GPU 6 Go+ (TensorFlow GPU) |
| Stockage | 5 Go (ChestMNIST) | 50 Go (NIH complet) |
