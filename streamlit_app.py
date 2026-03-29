"""
Démonstrateur applicatif — Système d'aide au tri radiologique
Streamlit app : EfficientNetV2S + ConvVAE + analyse textuelle multimodale
"""

import json
import marshal
import base64
import zipfile
import tempfile
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

import tensorflow as tf
import keras
from keras import layers, Model

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

CHESTMNIST_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

DEFAULT_THRESHOLD = 0.35

# ──────────────────────────────────────────────────────────────────────────────
# CSS personnalisé
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b3a5c 50%, #0d1b2a 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
}

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.15rem;
}

.status-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    color: #166534;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid #bbf7d0;
}
.status-err {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #fef2f2;
    color: #991b1b;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid #fecaca;
}

.result-detected {
    background: #fef2f2;
    border: 1px solid #fca5a5;
    border-left: 4px solid #ef4444;
    padding: 0.8rem 1.1rem;
    border-radius: 8px;
    color: #7f1d1d;
    font-weight: 500;
    margin-bottom: 1rem;
}
.result-normal {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid #22c55e;
    padding: 0.8rem 1.1rem;
    border-radius: 8px;
    color: #14532d;
    font-weight: 500;
    margin-bottom: 1rem;
}

.section-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    margin-bottom: 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e2e8f0;
}

.disclaimer {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: #78350f;
    margin-top: 1rem;
}

div[data-testid="stSidebar"] {
    background: #0d1b2a;
}
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] .stMarkdown,
div[data-testid="stSidebar"] span {
    color: #cbd5e1 !important;
}
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}
div[data-testid="stSidebar"] hr {
    border-color: #1e3a5f;
}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Architecture ConvVAE — enregistrée sous le nom utilisé lors de la sauvegarde
# ──────────────────────────────────────────────────────────────────────────────

class _SamplingLayer(layers.Layer):
    def call(self, inputs, training=None):
        mu, logvar = inputs
        if not training:
            return mu
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + eps * std


def _build_decoder(latent_dim: int, spatial_dim: int = 4) -> Model:
    inp = layers.Input(shape=(latent_dim,))
    x = layers.Dense(256 * spatial_dim * spatial_dim, activation='relu')(inp)
    x = layers.Reshape((spatial_dim, spatial_dim, 256))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    out = layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh')(x)
    return Model(inp, out, name='decoder')


@keras.saving.register_keras_serializable(package='custom', name='ConvVAE')
class ConvVAE(Model):
    """VAE convolutionnel — architecture identique au notebook 03, enregistrée pour le chargement."""

    def __init__(self, img_size: int = 64, latent_dim: int = 128, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.img_size   = img_size
        self.latent_dim = latent_dim
        self.beta       = beta

        enc_inp = layers.Input(shape=(img_size, img_size, 1))
        h = layers.Conv2D(32,  4, strides=2, padding='same', use_bias=False)(enc_inp)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Conv2D(64,  4, strides=2, padding='same', use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU(0.2)(h)
        h = layers.Flatten()(h)
        mu     = layers.Dense(latent_dim, name='mu')(h)
        logvar = layers.Dense(latent_dim, name='logvar')(h)
        self.encoder  = Model(enc_inp, [mu, logvar], name='encoder')
        self.sampling = _SamplingLayer()
        self.decoder  = _build_decoder(latent_dim)

    def call(self, x, training=None):
        mu, logvar = self.encoder(x, training=training)
        z          = self.sampling([mu, logvar], training=False)
        recon      = self.decoder(z, training=training)
        return recon, mu, logvar

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'img_size': self.img_size, 'latent_dim': self.latent_dim, 'beta': self.beta})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des modèles
# ──────────────────────────────────────────────────────────────────────────────

def _patch_keras_file(src_path: Path) -> str:
    """
    Retourne un fichier .keras patché en mémoire (chemin temporaire) si le fichier
    contient des Lambda layers avec du bytecode incompatible (Python version mismatch).
    Le patch remplace le bytecode par une identité Python 3.10.
    """
    identity_lambda = lambda imgs: imgs
    identity_code_b64 = base64.b64encode(marshal.dumps(identity_lambda.__code__)).decode()

    with zipfile.ZipFile(str(src_path), 'r') as zf:
        cfg = json.loads(zf.read('config.json'))
        weights_bytes = zf.read('model.weights.h5')
        meta_bytes    = zf.read('metadata.json')

    def patch_obj(obj):
        if isinstance(obj, dict):
            if obj.get('class_name') == '__lambda__':
                obj = dict(obj)
                obj['config'] = dict(obj.get('config', {}))
                obj['config']['code'] = identity_code_b64
                return obj
            return {k: patch_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [patch_obj(i) for i in obj]
        return obj

    cfg_patched = patch_obj(cfg)

    tmp = tempfile.NamedTemporaryFile(suffix='.keras', delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, 'w', compression=zipfile.ZIP_STORED) as zf_out:
        zf_out.writestr('config.json', json.dumps(cfg_patched))
        zf_out.writestr('model.weights.h5', weights_bytes)
        zf_out.writestr('metadata.json', meta_bytes)
    return tmp.name


@st.cache_resource
def load_classifier():
    model_path = MODELS_DIR / 'best_supervised_model.keras'
    if not model_path.exists():
        return None, "Modèle supervisé introuvable — lancez d'abord le notebook 02."

    keras.config.enable_unsafe_deserialization()

    # essai 1 : chargement direct
    try:
        model = keras.models.load_model(str(model_path), compile=False)
        return model, None
    except Exception:
        pass

    # essai 2 : patch du bytecode Lambda (Python version mismatch)
    try:
        patched = _patch_keras_file(model_path)
        model = keras.models.load_model(patched, compile=False)
        try:
            os.unlink(patched)
        except Exception:
            pass
        return model, None
    except Exception as e:
        return None, f"Impossible de charger le classifieur ({type(e).__name__}: {str(e)[:120]})"


@st.cache_resource
def load_vae(img_size: int = 64):
    vae_path = MODELS_DIR / 'ConvVAE_best.keras'
    if not vae_path.exists():
        return None

    keras.config.enable_unsafe_deserialization()

    # essai 1 : direct (ConvVAE enregistré)
    try:
        vae = keras.models.load_model(str(vae_path), compile=False,
                                       custom_objects={'ConvVAE': ConvVAE})
        return vae
    except Exception:
        pass

    # essai 2 : reconstruction + chargement des poids depuis l'archive
    try:
        vae = ConvVAE(img_size=img_size, latent_dim=128)
        vae(np.zeros((1, img_size, img_size, 1), dtype=np.float32))

        with zipfile.ZipFile(str(vae_path), 'r') as zf:
            with tempfile.TemporaryDirectory() as tmp:
                zf.extract('model.weights.h5', tmp)
                vae.load_weights(os.path.join(tmp, 'model.weights.h5'))
        return vae
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Prétraitements
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_for_classifier(pil_img: Image.Image, use_rgb: bool = True) -> np.ndarray:
    img = pil_img.convert('L').resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    if use_rgb:
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = arr[:, :, np.newaxis]
    return arr[np.newaxis, ...]


def preprocess_for_vae(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert('L').resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return arr[np.newaxis, :, :, np.newaxis]


# ──────────────────────────────────────────────────────────────────────────────
# Inférence
# ──────────────────────────────────────────────────────────────────────────────

def predict(model, img_tensor: np.ndarray, threshold: float):
    logits = model(img_tensor, training=False)
    probs  = tf.sigmoid(logits).numpy().squeeze()
    preds  = {cls: float(p) for cls, p in zip(CHESTMNIST_CLASSES, probs)}
    return preds, [cls for cls, p in preds.items() if p >= threshold]


def anomaly_score(vae, img_tensor: np.ndarray):
    img_tf = tf.constant(img_tensor, dtype=tf.float32)
    recon, mu, logvar = vae(img_tf, training=False)
    mse = float(tf.reduce_mean(tf.square(img_tf - recon)).numpy())
    kl  = float(-0.5 * tf.reduce_mean(1.0 + logvar - tf.square(mu) - tf.exp(logvar)).numpy())
    recon_np = (recon.numpy().squeeze() * 0.5 + 0.5).clip(0, 1)
    return mse + kl, recon_np


# ──────────────────────────────────────────────────────────────────────────────
# Graphiques
# ──────────────────────────────────────────────────────────────────────────────

def plot_proba_bars(preds: dict, threshold: float) -> plt.Figure:
    sorted_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    classes = [k for k, _ in sorted_items]
    probs   = [v for _, v in sorted_items]
    colors  = ['#ef4444' if p >= threshold else '#3b82f6' for p in probs]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    bars = ax.barh(classes, probs, color=colors, height=0.6, edgecolor='none')
    ax.axvline(x=threshold, color='#1e293b', linestyle='--', lw=1.2, alpha=0.7,
               label=f'Seuil ({threshold:.2f})')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité', fontsize=9, color='#475569')
    ax.tick_params(axis='both', labelsize=8.5, colors='#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.invert_yaxis()

    det_patch = mpatches.Patch(color='#ef4444', label='Détecté')
    neg_patch = mpatches.Patch(color='#3b82f6', label='Non détecté')
    ax.legend(handles=[det_patch, neg_patch, ax.get_lines()[0]],
              fontsize=8, framealpha=0.9, loc='lower right')
    ax.set_title('Probabilité par pathologie', fontsize=10, color='#0f172a', pad=10)
    plt.tight_layout()
    return fig


def plot_reconstruction(orig_np: np.ndarray, recon_np: np.ndarray) -> plt.Figure:
    error = np.abs(orig_np - recon_np)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.patch.set_facecolor('#ffffff')

    for ax, img, title, cmap in [
        (axes[0], orig_np,  'Original',          'gray'),
        (axes[1], recon_np, 'Reconstruction VAE', 'gray'),
        (axes[2], error,    "Carte d'erreur",     'hot'),
    ]:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, color='#334155', pad=6)
        ax.axis('off')

    plt.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout(pad=1.2)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Interface principale
# ──────────────────────────────────────────────────────────────────────────────

def sidebar_controls(classifier, vae):
    with st.sidebar:
        st.markdown("## ⚙️ Paramètres")
        threshold = st.slider(
            'Seuil de décision',
            min_value=0.10, max_value=0.80, value=DEFAULT_THRESHOLD, step=0.05,
            help='Probabilité minimale pour déclarer une pathologie détectée.'
        )
        show_vae = st.checkbox(
            'Analyse anomalie (VAE)',
            value=(vae is not None),
            disabled=(vae is None)
        )

        st.markdown("---")
        st.markdown("**Modèles chargés**")

        if classifier:
            st.markdown('<span class="status-ok">✓ Classifieur</span>', unsafe_allow_html=True)
            st.caption(f'`{classifier.name}`')
        else:
            st.markdown('<span class="status-err">✗ Classifieur</span>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        if vae:
            st.markdown('<span class="status-ok">✓ VAE anomalie</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">✗ VAE</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Dataset**")
        st.caption("ChestMNIST 64×64 · 14 classes · NIH ChestX-ray14")

    return threshold, show_vae


def section_mlflow_results():
    with st.expander("📊 Résultats d'entraînement (MLflow)", expanded=False):
        cols = st.columns(4)
        metrics = [
            ("EfficientNetV2-S ✅", "AUC 0.7684", "PR-AUC 0.158 · F1 0.168", "#22c55e"),
            ("DenseNet-121 (TL)",   "AUC 0.506",  "≈ hasard",                 "#f59e0b"),
            ("CNN scratch",         "AUC ~0.50",  "≈ hasard",                 "#94a3b8"),
            ("ConvVAE anomalie",    "AUC 0.52",   "légèrement > hasard",      "#94a3b8"),
        ]
        for col, (name, val, sub, color) in zip(cols, metrics):
            col.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{color}">{val}</div>
                <div class="label">{name}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.caption(
            "Seul EfficientNetV2-S apprend réellement — les autres restent proches du hasard sur "
            "ChestMNIST 64×64. Les modèles multimodaux affichent AUC 1.000 (fuite de données : "
            "le texte était généré depuis les labels eux-mêmes)."
        )


def section_upload():
    st.markdown('<div class="section-title">Radiographie</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        'Glissez une image ou cliquez pour parcourir',
        type=['png', 'jpg', 'jpeg'],
        label_visibility='collapsed'
    )
    pil_img = None
    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img.convert('L'), caption='Image chargée', use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Compte-rendu (optionnel)</div>',
                unsafe_allow_html=True)
    text = st.text_area(
        'Observations',
        height=90,
        placeholder='Ex : opacité basale droite évocatrice de consolidation…',
        label_visibility='collapsed'
    )
    return uploaded, pil_img, text


def section_results(classifier, vae, pil_img, threshold, show_vae, clinical_text):
    if pil_img is None:
        st.info('Chargez une radiographie dans le panneau de gauche pour lancer l\'analyse.')
        return

    if classifier is None:
        st.error('Classifieur non disponible — vérifiez que le modèle est présent dans `models/`.')
        return

    with st.spinner('Analyse en cours…'):
        input_shape = classifier.input_shape
        use_rgb = (input_shape[-1] == 3) if isinstance(input_shape, tuple) else True
        img_tensor = preprocess_for_classifier(pil_img, use_rgb=use_rgb)
        preds, detected = predict(classifier, img_tensor, threshold)

    if detected:
        st.markdown(
            f'<div class="result-detected">🔴 {len(detected)} pathologie(s) détectée(s) : '
            f'<strong>{", ".join(detected)}</strong></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-normal">🟢 Aucune pathologie détectée au seuil sélectionné</div>',
            unsafe_allow_html=True
        )

    st.pyplot(plot_proba_bars(preds, threshold), use_container_width=True)

    if show_vae and vae is not None:
        st.markdown('<div class="section-title" style="margin-top:1rem">Analyse d\'anomalie — VAE</div>',
                    unsafe_allow_html=True)
        vae_tensor = preprocess_for_vae(pil_img)
        score, recon_np = anomaly_score(vae, vae_tensor)

        orig_display = (vae_tensor.squeeze() * 0.5 + 0.5).clip(0, 1)
        high = score > 0.05

        c1, c2 = st.columns(2)
        with c1:
            st.metric('Score d\'anomalie', f'{score:.4f}',
                      delta='Atypique' if high else 'Normal',
                      delta_color='inverse')
        with c2:
            if high:
                st.markdown('🔴 **Image atypique** — vérification recommandée')
            else:
                st.markdown('🟢 **Dans la distribution normale**')

        st.pyplot(plot_reconstruction(orig_display, recon_np), use_container_width=True)

    if clinical_text.strip():
        st.markdown('<div class="section-title" style="margin-top:1rem">Analyse textuelle</div>',
                    unsafe_allow_html=True)
        st.warning(
            '⚠️ **Risque de fuite de données (data leakage).** '
            'En expérimentation, le texte était synthétisé depuis les labels — '
            'AUC artificiel de 1.000. Avec un vrai compte-rendu, AUC ≈ 0.53.',
            icon=None
        )
        kw = [c for c in CHESTMNIST_CLASSES
              if any(w in clinical_text.lower()
                     for w in c.lower().replace('_', ' ').split())]
        if kw:
            concordant  = set(detected) & set(kw)
            discordant  = set(detected).symmetric_difference(set(kw))
            st.write(f'**Mots-clés reconnus :** {", ".join(kw)}')
            if concordant:
                st.success(f'Concordance image / texte : {", ".join(concordant)}')
            if discordant:
                st.warning(f'Discordance : {", ".join(discordant)}')
        else:
            st.info('Aucun mot-clé pathologique reconnu dans le texte saisi.')


def main():
    st.set_page_config(
        page_title='Aide au tri radiologique',
        page_icon='🫁',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # chargement des modèles
    classifier, clf_err = load_classifier()
    vae = load_vae()

    # en-tête
    st.markdown("""
    <div class="main-header">
        <h1>🫁 Aide au tri radiologique</h1>
        <p>Prototype académique — classification de pathologies thoraciques via deep learning (Keras / TensorFlow)</p>
    </div>
    """, unsafe_allow_html=True)

    if clf_err:
        st.warning(clf_err)

    # contrôles sidebar
    threshold, show_vae = sidebar_controls(classifier, vae)

    # résultats MLflow (réduit par défaut)
    section_mlflow_results()

    st.markdown('<br>', unsafe_allow_html=True)

    # colonnes principales
    col_left, col_right = st.columns([1, 1.8], gap='large')

    with col_left:
        uploaded, pil_img, clinical_text = section_upload()

    with col_right:
        st.markdown('<div class="section-title">Résultats de l\'analyse</div>',
                    unsafe_allow_html=True)
        section_results(classifier, vae, pil_img, threshold, show_vae, clinical_text)

    # disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Avertissement :</strong> prototype académique conçu à des fins pédagogiques uniquement.
        Ne constitue pas un dispositif médical certifié et ne doit pas être utilisé pour un diagnostic clinique.
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
