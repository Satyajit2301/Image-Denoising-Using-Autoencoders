# 🧹 Image Denoising Using Autoencoders

An end-to-end deep learning project that implements and compares **five different autoencoder architectures** for removing Gaussian noise from handwritten digit images (MNIST dataset).

## 🌐 Live Demo

**[🚀 Try the Streamlit App](https://image-denoising-using-autoencoders-csnarvbtz8rkbrnazztpov.streamlit.app/)**

Upload or select a noisy MNIST digit, adjust the noise level, and watch the Convolutional Autoencoder clean it up in real time — with MSE, MAE, PSNR, and SSIM metrics displayed.

---

## 📌 Project Overview

Real-world images often contain noise or corruption that degrades their quality. This project deliberately corrupts MNIST images with Gaussian noise at varying levels (0.1–0.5) and trains autoencoders to reconstruct the clean originals.

### Denoising Workflow

```
Clean Image → Add Noise → Noisy Image → Encoder → Latent Space → Decoder → Cleaned Image
```

---

## 🏗️ Autoencoder Architectures Implemented

| # | Model | Core Idea | Parameters |
|---|-------|-----------|------------|
| 1 | **Undercomplete AE** | Latent dim (32) << Input dim (784) — forces compression | 476,720 |
| 2 | **Sparse AE** | L1 activity regularization on latent layer | 476,720 |
| 3 | **Denoising AE** | Dropout layers in encoder for additional noise robustness | 476,720 |
| 4 | **Convolutional AE** | Conv2D encoder + UpSampling decoder with BatchNorm | 75,265 |
| 5 | **Contractive AE** | Frobenius norm of encoder Jacobian as penalty term | 476,720 |

---

## 📊 Results Summary (Noise Factor = 0.3)

| Model | MSE ↓ | MAE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|-------|-------|-------|-------------|--------|
| Undercomplete AE | 0.01123 | 0.03621 | 19.50 | 0.8668 |
| Sparse AE | 0.02813 | 0.07159 | 15.51 | 0.6715 |
| Denoising AE | 0.01338 | 0.04350 | 18.74 | 0.8344 |
| **Convolutional AE** 🏆 | **0.00470** | **0.02336** | **23.28** | **0.9309** |
| Contractive AE | 0.01464 | 0.04394 | 18.34 | 0.8268 |
| *Noisy (Baseline)* | *0.04663* | *0.12806* | *13.31* | *0.5279* |

**🏆 Best Model: Convolutional Autoencoder** — lowest error, highest structural similarity, and only 75K parameters.

---

## 🔬 Key Experiments

### Noise-Level Analysis
- All models evaluated at noise factors 0.1, 0.2, 0.3, 0.4, and 0.5
- Convolutional AE maintains superiority across all noise levels
- All models degrade significantly at NF ≥ 0.5

### Latent-Dimension Analysis
- Tested latent dimensions: 8, 16, 32, 64, 128
- Larger latent dimensions improve quality with diminishing returns beyond 64
- Sparse and Contractive AEs struggle at very small latent dims (8, 16)

---

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** — model building and training
- **NumPy** — numerical operations
- **Matplotlib** — visualizations
- **scikit-image** — SSIM metric computation
- **Streamlit** — web app deployment

---

## 🚀 Deployment

The best-performing model (Convolutional AE) is deployed as a Streamlit web app.

### Features
- Select digit class or pick a random MNIST test image
- Adjustable noise factor slider (0.0 – 1.0)
- Side-by-side comparison: Original → Noisy → Denoised
- Real-time MSE, MAE, PSNR, SSIM metrics
- Download denoised images

### Run Locally

```bash
git clone https://github.com/Satyajit2301/Image-Denoising-Using-Autoencoders.git
cd Image-Denoising-Using-Autoencoders
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── app.py                              # Streamlit deployment app
├── requirements.txt                    # Python dependencies
├── best_model_convolutional_ae.keras   # Trained Convolutional AE model
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## 📝 Evaluation Metrics

| Metric | Interpretation | Preferred Direction |
|--------|---------------|-------------------|
| **MSE** | Average squared reconstruction error | Lower ↓ |
| **MAE** | Average absolute reconstruction error | Lower ↓ |
| **PSNR** | Signal-to-noise quality measure (dB) | Higher ↑ |
| **SSIM** | Structural similarity with clean image | Higher ↑ |

---

## 🏆 Conclusions

1. **Convolutional AE** achieves the best reconstruction quality with the fewest parameters (75K vs 477K for dense models)
2. Spatial feature extraction via Conv2D layers preserves edges and digit shapes far better than fully-connected architectures
3. Increasing latent dimensions improves quality but with diminishing returns beyond 64
4. All autoencoder models significantly outperform the noisy baseline across all metrics

---

## 📄 License

This project is developed as part of a laboratory assignment for academic purposes.
