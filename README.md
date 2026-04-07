# Q-Trust AI 🔬⚛️

### Hybrid Architecture for Synthetic Image Detection via Dual-Input Quantum Re-Uploading

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PennyLane-0.36+-black?style=for-the-badge&logo=data:image/svg+xml;base64,&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Quantum-6%20Qubits-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-90.75%25-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <b>Brazil Quantum Camp - Quantum Computing Solutions</b><br/>
  <i>Team Q-Trust AI</i>
</p>

---

## 📌 Overview

As AI-generated imagery becomes indistinguishable from real photographs, the need for robust detection methods is critical. Q-Trust AI presents a **hybrid classical-quantum architecture** that fuses deep visual features and spectral frequency signatures to classify images as real or synthetic.

Our key insight: generative models like GANs and Diffusion Models leave measurable **spectral fingerprints** - patterns invisible to the human eye but detectable in the frequency domain. By combining these spectral features with ResNet18 semantic embeddings inside a **Variational Quantum Circuit (VQC)**, we exploit quantum entanglement to capture subtle cross-modal correlations.

> **Result:** 90.75% accuracy and F1-Score of 0.9125 on the CIFAKE benchmark - operating in only **6 dimensions** with **72 trainable quantum parameters**, near-matching a classical SVM with 528 dimensions.

---

## 🧠 Architecture

```
Input Image (32×32×3)
    │
    ├──► ResNet18 (ImageNet, frozen + fine-tuned layer4)
    │         └──► 512D ──► Linear(512→6) ──► tanh×(π/2) ──► semantic_enc (6D)
    │
    └──► Radial FFT Power Spectrum (16 bins)
              └──► Linear(16→6) ──► tanh×(π/2) ──► spectral_enc (6D)
                                                          │
                                    ┌─────────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │   Variational Quantum Circuit    │
                    │   6 qubits · 4 Re-Upload layers │
                    │                                  │
                    │  RY(semantic[i]) ─ RZ(spectral[i]) per qubit  │
                    │  CNOT ring entanglement          │
                    └──────────────┬──────────────────┘
                                   │
                          ⟨Z₀⟩ ... ⟨Z₅⟩  (6 Pauli-Z expectation values)
                                   │
                              Linear(6→2) ──► Softmax ──► P(fake)
```


### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Visual Encoder** | ResNet18 (partial fine-tuning) | Rich 512D semantic features; transfer learning from ImageNet |
| **Spectral Encoder** | Radial FFT (16 bins, 32×32 resolution) | Captures GAN/diffusion sampling artifacts with full frequency granularity |
| **Embedding** | Orthogonal Angle Embedding (RY + RZ) | Semantic and spectral features occupy separate Bloch sphere axes - no destructive interference |
| **Entanglement** | CNOT ring | Propagates cross-modal correlations across all qubits |
| **Re-Uploading** | 4 layers of Data Re-Uploading | Transforms shallow VQC into a universal function approximator |
| **Optimizer** | AdamW + Label Smoothing (ε=0.05) | Stabilizes classical-quantum gradient interface; prevents overconfidence |

---

## 📊 Results

### Test Set Performance (400 images, never seen during training)

| Model | Accuracy | F1-Score | AUC-ROC | Params (quantum) | Input Dim |
|-------|----------|----------|---------|-----------------|-----------|
| **VQC Dual-Input (ours)** | **90.75%** | **0.9125** | **0.9563** | **72** | **6** |
| MLP Equivalent (6D) | 90.75% | 0.9082 | - | ~50 | 6 |
| Logistic Regression | 90.25% | - | - | - | 528 |
| SVM RBF (upper bound) | 91.00% | 0.9167 | - | - | 528 |

> **The VQC with 6D input outperforms Logistic Regression trained on 528 dimensions**, and ties the classical equivalent MLP on accuracy while surpassing it on F1-Score - suggesting the advantage comes from quantum expressivity via entanglement, not raw computational power.

### Confusion Matrix Highlights

- **Synthetic recall: 96%** - the model catches almost all fake images
- **Real precision: 85%** - conservative bias ideal for anti-fraud systems
- AUC-ROC of **0.9563** demonstrates robust discrimination across all thresholds

### t-SNE Analysis

The 6 Pauli-Z observables produce **clearer geometric separation** between real and fake classes compared to the classical 6D equivalent, visually confirming the advantage of orthogonal multimodal encoding in quantum latent space.

---

## 🗂️ Repository Structure

```
q-trust-ai/
│
├── README.md                          # This file
├── LICENSE
│
├── Q_Trust_AI_notebook.ipynb          # Full experiment notebook (Google Colab)
│
├── docs/
│   ├── ARCHITECTURE.md                # Detailed architecture walkthrough
│   ├── EXPERIMENTS.md                 # Full experiment log (all tested architectures)
│   └── final_report.pdf               # Official competition report (PT-BR)
│
└── assets/
    └── architecture_diagram.png       # Architecture overview figure
```

---

## ⚙️ Setup & Reproduction

### Requirements

```bash
pip install pennylane pennylane-lightning scikit-learn matplotlib seaborn kagglehub tqdm
pip install torch torchvision
```

### Running the Notebook

The full experiment is contained in a single Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qmERT8cg17a1OMGrXEX7T_PYX_Xa1MnP?usp=sharing)

All global constants controlling the experiment are defined at the top of Section 1. Key reproducibility parameters:

```python
SEED        = 42       # Fixed across numpy, torch, random
N_QUBITS    = 6        # Qubits in the VQC
N_LAYERS    = 4        # Data Re-Uploading layers
TRAIN_SIZE  = 2000     # Balanced: 1000 REAL + 1000 FAKE
VAL_SIZE    = 400      # 200 + 200
TEST_SIZE   = 400      # 200 + 200 (isolated)
```

### Dataset

We use the **CIFAKE** benchmark ([Bird & Lotfi, 2024](https://arxiv.org/abs/2303.14126)), automatically downloaded via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("bird-j/cifake-real-and-ai-generated-synthetic-images")
```

120,000 images of 32×32 pixels: CIFAR-10 real photographs vs. Stable Diffusion 1.4 synthetic counterparts.

---

## 🔬 Sections of the Notebook

| Section | Description |
|---------|-------------|
| **1 - Setup & Imports** | Dependencies, global constants, reproducibility seed |
| **2 - Dataset & FFT Features** | CIFAKE loading, radial power spectrum extraction |
| **3 - Exploratory Analysis** | Visual samples, spectral signature comparison (real vs. fake) |
| **4 - Architecture** | VQC circuit definition, ResNet encoder, Angle Embedding |
| **5 - Hybrid Training** | Data augmentation, AdamW, Label Smoothing, training loop |
| **6 - Test Evaluation** | Accuracy, F1-Score, AUC-ROC, confusion matrix |
| **7 - Baseline Models** | SVM RBF, Logistic Regression, MLP 6D |
| **8 - Comparative Analysis** | ROC curves, precision-recall, probability distributions |
| **9 - Quantum Circuit Internals** | Pauli-Z observables, t-SNE of quantum latent space |
| **10 - Computational Complexity** | P vs NP-hard framing; motivation for hybrid QML |
| **11 - Conclusions & Next Steps** | Full results table, lessons learned, future directions |

---

## 🚀 Next Steps

| Direction | Description | Expected Impact |
|-----------|-------------|-----------------|
| **More Re-Uploading layers** | Increase `N_LAYERS` from 4 to 6–8 | Higher accuracy |
| **Strongly Entangling Layers** | Replace CNOT ring with denser entanglement | Better multivariate correlation capture |
| **Real quantum hardware** | Run on IBM Quantum or IonQ with SPSA optimizer | Assess decoherence impact on accuracy |
| **New generators** | Test against Midjourney, Flux2 (not just Stable Diffusion) | Generalization evaluation |
| **Quantum Natural Gradient** | Replace AdamW with QNG optimizer | More stable quantum parameter updates |
| **Larger datasets** | Full CIFAKE (120K) + other benchmarks | Scalability assessment |

---

## 👥 Team

**Q-Trust AI** - Brazil Quantum Camp 2026

Amanda Arruda · Caio Silva · Diogo Lacerda · Eduarda Mendes · Igor Oliveira · Paulo Aquino · Rebeca Vitória Tenório · Vinícius Leal

---

## 📚 References

1. Bird, J. J., & Lotfi, A. (2023). **CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images**. arXiv:2303.14126. [https://arxiv.org/abs/2303.14126](https://arxiv.org/abs/2303.14126)

2. Blum, A. L., & Rivest, R. L. (1992). **Training a 3-node neural network is NP-complete**. *Neural Networks*, 5(1), 117-127.

3. Citron, D. K., & Chesney, R. (2019). **Deepfakes and the New Disinformation War**. Boston University School of Law.

4. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., et al. (2020). **Data re-uploading for a universal quantum classifier**. *Quantum*, 4, 226. [https://arxiv.org/abs/1907.02085](https://arxiv.org/abs/1907.02085)

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
