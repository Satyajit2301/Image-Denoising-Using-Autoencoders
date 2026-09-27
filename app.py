"""
Streamlit deployment app — Lab Assignment 4: Image Denoising Using Autoencoders
--------------------------------------------------------------------------------
Lets you pick a random MNIST test-set digit, corrupt it with adjustable Gaussian
noise, and see the trained Convolutional Autoencoder (the best-performing model
in the notebook) clean it back up — with MSE / MAE / PSNR / SSIM metrics.

HOW TO USE YOUR OWN TRAINED MODEL (recommended — much faster & matches your
notebook's actual results exactly):
    1. In Colab, after running the notebook, download `best_model_convolutional_ae.keras`
       (it's already created in Block 14 / saved into the zip in Block 16).
    2. Put that file in the SAME folder as this app.py.
    3. Run the app — it will detect and load it automatically instead of retraining.

If that file isn't present, the app trains a lightweight version of the same
architecture from scratch on first run (a few minutes on CPU) and caches the
result to disk so every run after that is instant.

Run with:  streamlit run app.py
"""

import io
import os

import numpy as np
import streamlit as st
from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers, models
from skimage.metrics import structural_similarity as ssim

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="MNIST Denoising Autoencoder", page_icon="🧹", layout="centered")

MODEL_PATH = "best_model_convolutional_ae.keras"  # drop your Colab-trained file here to skip retraining
TRAIN_NOISE_FACTOR = 0.3   # same default noise level the notebook trained at
TRAIN_EPOCHS = 8           # kept small so a from-scratch run finishes in a couple of minutes on CPU
BATCH_SIZE = 256


# --------------------------------------------------------------------------- #
# Data & model
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_mnist():
    (x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    return x_train, x_test, y_test


def build_conv_ae():
    """Same 'v2' Convolutional Autoencoder architecture used in the notebook
    (Block 9 rebuild) — the model the lab's own evaluation ranked best
    (lowest MSE, highest PSNR/SSIM, 75,265 params)."""
    encoder_input = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)          # 14x14x32

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    latent = layers.MaxPooling2D((2, 2), padding="same")(x)     # 7x7x64

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(latent)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)                          # 14x14x64

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)                          # 28x28x32

    decoder_output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = models.Model(encoder_input, decoder_output, name="Convolutional_AE")
    model.compile(optimizer="adam", loss="mse")
    return model


@st.cache_resource(show_spinner=False)
def get_model():
    """Load a saved model if present, otherwise train one from scratch and cache it."""
    if os.path.exists(MODEL_PATH):
        return keras.models.load_model(MODEL_PATH), False

    x_train, x_test, _ = load_mnist()
    noisy_train = add_noise(x_train, TRAIN_NOISE_FACTOR, seed=42)
    noisy_test = add_noise(x_test, TRAIN_NOISE_FACTOR, seed=43)

    model = build_conv_ae()
    model.fit(
        noisy_train, x_train,
        epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(noisy_test, x_test),
        verbose=0,
    )
    model.save(MODEL_PATH)
    return model, True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def add_noise(image, noise_factor, seed=None):
    rng = np.random.default_rng(seed)
    noisy = image + noise_factor * rng.normal(size=image.shape)
    return np.clip(noisy, 0, 1).astype("float32")


def compute_metrics(original, reconstructed):
    mse = float(np.mean((original - reconstructed) ** 2))
    mae = float(np.mean(np.abs(original - reconstructed)))
    psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    ssim_val = float(ssim(original.squeeze(), reconstructed.squeeze(), data_range=1.0))
    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "SSIM": ssim_val}


def to_display_image(arr, size=180):
    img = Image.fromarray((arr.squeeze() * 255).astype(np.uint8), mode="L")
    return img.resize((size, size), Image.NEAREST)


def image_download_bytes(arr):
    img = Image.fromarray((arr.squeeze() * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pick_new_sample():
    _, x_test, y_test = load_mnist()
    choice = st.session_state.get("digit_choice", "Any")
    if choice == "Any":
        st.session_state.sample_idx = int(np.random.randint(len(x_test)))
    else:
        candidates = np.where(y_test == int(choice))[0]
        st.session_state.sample_idx = int(np.random.choice(candidates))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🧹 MNIST Denoising Autoencoder")
st.caption("Lab Assignment 4 — deployment demo of the Convolutional Autoencoder")

x_train, x_test, y_test = load_mnist()

with st.spinner("Preparing model... (first run without a saved .keras file trains one, ~1–2 min on CPU)"):
    model, freshly_trained = get_model()

if freshly_trained:
    st.info(
        f"No `{MODEL_PATH}` found, so a fresh copy was trained for {TRAIN_EPOCHS} epochs "
        f"and cached to disk. For results that match your notebook exactly, drop your "
        f"Colab-trained `{MODEL_PATH}` into this app's folder and restart."
    )
else:
    st.success(f"Loaded trained model from `{MODEL_PATH}`.")

st.divider()

col_a, col_b = st.columns([1, 2])
with col_a:
    st.selectbox(
        "Digit class", ["Any"] + [str(d) for d in range(10)],
        index=0, key="digit_choice", on_change=pick_new_sample,
    )
with col_b:
    noise_factor = st.slider("Noise factor", 0.0, 1.0, TRAIN_NOISE_FACTOR, 0.05)

if "sample_idx" not in st.session_state:
    pick_new_sample()

st.button("🎲 New random sample", on_click=pick_new_sample)

idx = st.session_state.sample_idx
original = x_test[idx]
noisy = add_noise(original, noise_factor, seed=idx * 10_000 + int(noise_factor * 1000))
denoised = model.predict(noisy.reshape(1, 28, 28, 1), verbose=0).reshape(28, 28, 1)

st.caption(f"Test-set index {idx} — true label **{y_test[idx]}**")
c1, c2, c3 = st.columns(3)
c1.image(to_display_image(original), caption="Original")
c2.image(to_display_image(noisy), caption=f"Noisy (factor={noise_factor:.2f})")
c3.image(to_display_image(denoised), caption="Denoised")

st.download_button(
    "⬇️ Download denoised image",
    data=image_download_bytes(denoised),
    file_name=f"denoised_digit_{y_test[idx]}_{idx}.png",
    mime="image/png",
)

st.divider()
st.subheader("Quantitative metrics (vs. the clean original)")

noisy_metrics = compute_metrics(original, noisy)
denoised_metrics = compute_metrics(original, denoised)

mcol1, mcol2 = st.columns(2)
with mcol1:
    st.markdown("**Noisy image**")
    st.metric("MSE", f"{noisy_metrics['MSE']:.5f}")
    st.metric("MAE", f"{noisy_metrics['MAE']:.5f}")
    st.metric("PSNR (dB)", f"{noisy_metrics['PSNR']:.2f}")
    st.metric("SSIM", f"{noisy_metrics['SSIM']:.4f}")
with mcol2:
    st.markdown("**Denoised image**")
    st.metric("MSE", f"{denoised_metrics['MSE']:.5f}",
               delta=f"{denoised_metrics['MSE'] - noisy_metrics['MSE']:.5f}", delta_color="inverse")
    st.metric("MAE", f"{denoised_metrics['MAE']:.5f}",
               delta=f"{denoised_metrics['MAE'] - noisy_metrics['MAE']:.5f}", delta_color="inverse")
    st.metric("PSNR (dB)", f"{denoised_metrics['PSNR']:.2f}",
               delta=f"{denoised_metrics['PSNR'] - noisy_metrics['PSNR']:.2f}")
    st.metric("SSIM", f"{denoised_metrics['SSIM']:.4f}",
               delta=f"{denoised_metrics['SSIM'] - noisy_metrics['SSIM']:.4f}")
