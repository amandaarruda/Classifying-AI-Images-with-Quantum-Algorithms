# Experiment Log

This document records all architectures and configurations tested during the Q-Trust AI project, including dead ends and lessons learned. The goal is full reproducibility and transparency about the path to the final architecture.

---

## Final Architecture — VQC Dual-Input Re-Uploading (6D)

**Result: 90.75% ACC · F1 0.9125 · AUC-ROC 0.9563**

The architecture described in [ARCHITECTURE.md](ARCHITECTURE.md). Dual-input ResNet18 + FFT features, orthogonal angle embedding, 4-layer Data Re-Uploading, CNOT ring entanglement.

---

## Baseline Models

These classical models were trained on the same data split using the same features extracted by the final architecture. They serve as upper and lower bounds for evaluating the VQC contribution.

### SVM RBF (528D) — Upper Bound

| Metric | Value |
|--------|-------|
| Accuracy | **91.00%** |
| F1-Score | 0.9167 |
| Input dim | 528 (ResNet 512D + FFT 16D) |
| Parameters | kernel-defined |

The absolute classical ceiling on this dataset and feature set. The VQC approaches this with 88× fewer input dimensions.

### Logistic Regression (528D)

| Metric | Value |
|--------|-------|
| Accuracy | 90.25% |
| Input dim | 528 |

Linear baseline. The VQC surpasses this despite operating in 6D — highlighting the non-linear expressivity of the quantum circuit.

### MLP Equivalent (6D) — Direct Classical Counterpart

| Metric | Value |
|--------|-------|
| Accuracy | 90.75% |
| F1-Score | 0.9082 |
| Input dim | 6 |
| Architecture | Linear(6→12) → ReLU → Linear(12→2) |

The most important comparison: a classical network with the **identical input** (6D compressed features) and similar parameter count. The VQC **ties on accuracy and surpasses on F1-Score (0.9125 vs 0.9082)**, suggesting quantum expressivity provides a measurable edge.

---

## Intermediate Architectures (Pre-final iterations)

These architectures were tested before arriving at the final design. Each represents a specific hypothesis and the lesson it taught the team.

### Iteration 1 — FFT Classical + PCA + VQC (8D)

**Result: 80.87% ACC**

**Configuration:**
- Input: FFT spectral features only (no ResNet)
- Dimensionality reduction: PCA to 8 dimensions
- VQC: 8 qubits, single-input angle embedding

**Hypothesis:** Raw spectral features alone are sufficient for synthetic detection.

**Outcome:** Failed to capture the complexity of visual artifacts. PCA compressed away too much information. The 8-qubit simulation was also significantly slower without proportional accuracy gains.

**Lesson:** Spectral features alone are insufficient — semantic visual context from deep features is necessary. The 8-qubit overhead wasn't justified.

---

### Iteration 2 — ResNet Only + VQC (6D)

**Result: ~87% ACC (internal validation)**

**Configuration:**
- Input: ResNet18 512D → Linear(512→6) only (no FFT branch)
- VQC: 6 qubits, single-input angle embedding (RY only)

**Hypothesis:** ResNet features alone, compressed to 6D, can provide enough signal for the VQC.

**Outcome:** Decent performance, but significant variance across runs. The model struggled to distinguish high-quality synthetic images that passed visual inspection.

**Lesson:** Visual features alone miss the frequency-domain artifacts. Adding FFT was the key step that pushed performance past the logistic regression baseline.

---

### Iteration 3 — Dual-Input, Single-Axis Embedding

**Result: ~88.5% ACC (internal validation)**

**Configuration:**
- Input: ResNet + FFT both encoded via RY (same axis)
- VQC: 6 qubits, 4 Re-Uploading layers

**Hypothesis:** Concatenating both modalities in the same rotation axis would be sufficient.

**Outcome:** Better than single-input, but noticeably worse than orthogonal embedding. Analysis of Pauli-Z observables showed partial interference between the two feature types.

**Lesson:** Encoding both modalities on the same rotation axis causes destructive interference in the Bloch sphere. The orthogonal RY+RZ embedding (visual on Y, spectral on Z) was the critical architectural choice that enabled clean coexistence of both signals in the quantum state.

---

### Iteration 4 — No Data Re-Uploading (single encoding)

**Result: ~84% ACC (internal validation)**

**Configuration:**
- Same as final architecture but with N_LAYERS = 1 (single Re-Uploading)
- Effectively a single feature encoding followed by entanglement

**Hypothesis:** One encoding pass with strong entanglement might be sufficient.

**Outcome:** Significant underfitting. The single-layer VQC behaved like a linear classifier in the quantum feature space — insufficient expressivity for this binary classification task.

**Lesson:** Data Re-Uploading is non-negotiable. Without it, the VQC cannot approximate the complex decision boundary separating real from synthetic images. Each additional layer adds higher-frequency components to the quantum Fourier decomposition.

---

## Ablation Summary

| Architecture | Accuracy | Key Change | Lesson |
|-------------|----------|------------|--------|
| FFT + PCA + VQC (8D) | 80.87% | Spectral only | Visual context required |
| ResNet-only + VQC (6D) | ~87% | No FFT | Spectral signatures matter |
| Dual-input, single axis (RY) | ~88.5% | No orthogonal embed | Axis orthogonality prevents interference |
| Single Re-Upload (N_LAYERS=1) | ~84% | No multi-layer | Re-Uploading essential for expressivity |
| **Final (RY+RZ, 4 layers)** | **90.75%** | Full architecture | All components synergize |

---

## Hyperparameter Sensitivity

| Parameter | Values Tested | Best | Notes |
|-----------|--------------|------|-------|
| N_QUBITS | 4, 6, 8 | **6** | 4 too limited; 8 sim cost too high |
| N_LAYERS | 1, 2, 4, 6 | **4** | Diminishing returns after 4; 6 was slower without gain |
| Learning rate | 1e-4, 3e-4, 1e-3 | **3e-4** | Standard AdamW sweet spot |
| Label smoothing ε | 0.0, 0.05, 0.1 | **0.05** | 0.1 hurt recall; 0.0 led to overconfident VQC |
| FFT bins | 8, 16, 32 | **16** | 32 introduced noise; 8 lost granularity |
| Compressed dim | 4, 6, 8 | **6** | Matches qubit count; 4 too lossy, 8 unnecessary |

---

## Dataset Notes

We used a **stratified subset** of CIFAKE rather than the full 120K dataset due to the computational cost of quantum circuit simulation:

- **Training:** 2,000 images (1,000 REAL + 1,000 FAKE)
- **Validation:** 400 images (200 + 200)
- **Test:** 400 images (200 + 200) — strictly isolated

Simulation time per epoch on a standard Colab GPU (T4): approximately 4–8 minutes depending on batch size. Full CIFAKE training would require hardware acceleration or a real quantum device.

**Data Augmentation was applied only to the visual (ResNet) branch**, not to FFT features, to preserve the integrity of spectral signatures. Augmenting the FFT input would corrupt the very artifacts we're trying to detect.
