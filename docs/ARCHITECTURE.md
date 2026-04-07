# Architecture Deep Dive: Dual-Input VQC with Quantum Re-Uploading

This document provides a detailed walkthrough of the Q-Trust AI hybrid architecture for synthetic image detection.

---

## Motivation

### Why Frequency Domain?

Generative models such as GANs and Diffusion Models introduce **measurable spectral artifacts** during the image synthesis process. These artifacts — visible only in the frequency domain — arise from:

- **Upsampling artifacts** in GAN decoders (checkerboard patterns)
- **Denoising step regularity** in Diffusion Models introducing periodic noise residuals
- **Convolutional bias** toward repeating textures at specific frequency bands

Classical CNNs operating purely in the spatial domain are generalist architectures — they do not explicitly model the frequency structure of images. Our architecture adds a dedicated FFT branch to exploit this signal.

### Why Quantum?

The optimal training of deep neural networks is **NP-hard** (Blum & Rivest, 1992): finding the global minimum in non-convex loss landscapes cannot be done in polynomial time. This asymmetry motivates hybrid Quantum Machine Learning (QML), where the high expressivity of Variational Quantum Circuits (VQCs) is leveraged at the most critical learning steps.

Specifically, VQCs operating via **quantum entanglement** can represent correlations that would require exponentially more parameters in a classical network. In our case, this allows the 6-qubit circuit to capture cross-modal interactions between visual semantics and spectral patterns.

---

## Component Breakdown

### 1. Visual Branch — ResNet18

```
Input image (3×32×32)
  └──► ResNet18 (pretrained on ImageNet)
          ├── layers 1–3: frozen
          └── layer 4: fine-tuned (last convolutional block)
                └──► Global Average Pooling
                        └──► 512D feature vector
                                └──► Linear(512 → 6) + tanh × (π/2)
                                        └──► semantic_enc ∈ [-π/2, π/2]⁶
```

**Why partial fine-tuning?** Freezing early layers preserves low-level ImageNet features (edges, textures). Fine-tuning layer 4 allows the network to adapt higher-level representations to the real/fake discrimination task without catastrophic forgetting and with far fewer parameters to update.

**Why tanh × (π/2)?** The output of the linear layer is compressed to `[-π/2, π/2]`, which maps naturally to rotation angles in the quantum circuit. This normalization prevents gradient explosion at the classical-quantum interface.

---

### 2. Spectral Branch — Radial FFT

```
Input image (3×32×32)
  └──► Convert to grayscale
          └──► 2D FFT
                └──► Shift zero-frequency to center
                        └──► Compute power spectrum: |F(u,v)|²
                                └──► Radial binning into 16 bins
                                        (from DC to Nyquist, 1px granularity at 32×32)
                                              └──► Log normalization
                                                      └──► Linear(16 → 6) + tanh × (π/2)
                                                              └──► spectral_enc ∈ [-π/2, π/2]⁶
```

**Why radial binning?** The frequency content of images is isotropic in direction but highly informative along the radial axis (low → high frequency). 16 radial bins at 32×32 native resolution preserve full frequency granularity without dimensionality explosion.

**Why 16 bins at 32×32?** The maximum spatial frequency in a 32×32 image is 16 cycles/pixel (Nyquist limit), so 16 bins provide exactly 1 bin per pixel frequency — capturing every detectable artifact without losing information.

---

### 3. Variational Quantum Circuit (VQC)

#### Circuit Structure

```
For each of 4 Re-Uploading layers:
  For each qubit i (0 to 5):
    RY(semantic_enc[i])    ← visual rotation (Bloch Y-axis)
    RZ(spectral_enc[i])    ← spectral rotation (Bloch Z-axis)
  
  CNOT ring entanglement:
    CNOT(0→1), CNOT(1→2), CNOT(2→3), CNOT(3→4), CNOT(4→5), CNOT(5→0)

Output: ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩, ⟨Z₄⟩, ⟨Z₅⟩  (Pauli-Z expectation values)
```

#### Orthogonal Angle Embedding

The core architectural innovation is encoding the two modalities on **orthogonal rotation axes** of the Bloch sphere:

- **RY gates** ← ResNet semantic features (visual content)
- **RZ gates** ← FFT spectral features (frequency patterns)

In the Bloch sphere geometry, RY and RZ rotations are orthogonal transformations. This means:

1. Each feature type controls an independent degree of freedom per qubit
2. No destructive interference between the two modalities
3. The quantum state simultaneously encodes both signals without information mixing

This is analogous to using orthogonal basis vectors in a classical representation — but in the exponentially large Hilbert space of quantum states.

#### Data Re-Uploading

Introduced by Pérez-Salinas et al. (2020), Data Re-Uploading is the technique of encoding input features **multiple times** (once per layer) rather than a single initialization. This transforms a shallow VQC into a **universal function approximator**.

Without Re-Uploading, a 4-layer VQC with fixed input encoding would be equivalent to a linear model in the quantum feature space. With Re-Uploading, each layer re-encodes the inputs with learned rotation offsets, exponentially increasing the model's expressivity.

**Mathematical intuition:** The circuit implements a Fourier-like decomposition where each Re-Uploading layer adds higher-frequency components to the learned function, enabling approximation of arbitrarily complex decision boundaries.

#### CNOT Ring Entanglement

After each layer of single-qubit rotations, a ring of CNOT gates connects adjacent qubits:

```
q0 ──●──────────────────────X──
     │                      │
q1 ──X──●───────────────────┼──
         │                  │
q2 ──────X──●───────────────┼──
             │              │
q3 ──────────X──●───────────┼──
                 │          │
q4 ──────────────X──●───────┼──
                     │      │
q5 ──────────────────X──────●──
```

This creates **multi-qubit entanglement**: the quantum state of each qubit becomes correlated with all others. In the context of our architecture, this propagates cross-modal correlations — the semantic encoding of qubit 0 influences the measurement of qubit 5, creating joint representations that classical architectures cannot replicate with equivalent parameters.

---

### 4. Classification Head

```
⟨Z₀⟩, ..., ⟨Z₅⟩  ∈ [-1, 1]⁶
        │
   Linear(6 → 2)
        │
     Softmax
        │
   P(real), P(fake)
```

The 6 Pauli-Z expectation values are each bounded in `[-1, 1]`. A final linear layer projects this to 2 logits, and softmax converts to probabilities. The classification threshold is 0.5 on `P(fake)`.

---

## Training Strategy

### Data Augmentation (Visual Branch Only)

Applied exclusively to the ResNet input to prevent early memorization:

```python
transforms.RandomResizedCrop(32, scale=(0.8, 1.0))
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
```

FFT features are computed from the **original, unaugmented** image to preserve spectral integrity.

### Label Smoothing (ε = 0.05)

Instead of hard labels {0, 1}, the model is trained with soft targets {0.025, 0.975}. This:
- Penalizes overconfidence
- Forces the circuit to learn generalizable features rather than memorizing training noise
- Particularly important at the quantum-classical interface where gradient signals can be sharp

### AdamW Optimizer

Standard Adam accumulates squared gradients, which can cause issues at the quantum-classical boundary where gradient magnitudes differ significantly. AdamW applies **decoupled weight decay**, stabilizing the gradient flow and preventing parameter explosion in the classical layers adjacent to the VQC.

---

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| ResNet18 (fine-tuned layer 4) | ~2.1M |
| Linear(512→6) | 3,078 |
| Linear(16→6) | 102 |
| VQC (6 qubits × 2 rotations × 4 layers) | **72** |
| Linear(6→2) | 14 |
| **Total (active training)** | **~2.1M** |
| **Quantum parameters only** | **72** |

The quantum component uses only 72 parameters — yet its contribution to the final classification is what distinguishes this architecture from its classical equivalent.

---

## Implementation Details

**Framework:** PyTorch (classical components) + PennyLane with `lightning.qubit` device (quantum simulation)

**Quantum device:**
```python
dev = qml.device("lightning.qubit", wires=N_QUBITS)
```

**Gradient computation:** Parameter-shift rule for quantum gradients, automatic differentiation for classical components. PennyLane's hybrid autograd handles the classical-quantum gradient interface transparently.

**Simulation cost:** Simulating a 6-qubit statevector requires tracking 2⁶ = 64 complex amplitudes. With 4 Re-Uploading layers and a ring of 6 CNOTs per layer, the circuit depth is manageable for classical simulation while still providing meaningful entanglement.
