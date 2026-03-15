# Latent Consensus Manifold

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Latent-Consensus-Manifold?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Latent-Consensus-Manifold?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Multi-model consensus in latent space — measuring agreement, divergence, and structural alignment between representations learned by different neural network architectures.

---

**Topics:** `abstract-reasoning` · `cognitive-architecture` · `counterfactual-causal-verification` · `mdl-scored-hypothesis` · `multi-agent-systems` · `neuro-symbolic-reasoning` · `popperian-falsification` · `procedural-reasoning` · `socratic-debate`

## Overview

Latent Consensus Manifold investigates a fundamental question in representation learning: when different
neural network architectures (ResNet, ViT, EfficientNet, MLP-Mixer) are trained on the same dataset,
do they learn the same latent representations, or fundamentally different ones? The answer has
profound implications for model distillation, ensemble construction, transfer learning, and our
theoretical understanding of what neural networks are computing.

The project implements representational similarity analysis (RSA) and Centered Kernel Alignment (CKA)
to measure the geometric similarity between representation spaces of different models — providing
a quantitative measure of consensus between architectures. CKA, in particular, is invariant to
orthogonal transformation and isotropic scaling of representations, making it a principled distance
measure that correctly identifies similar functional representations regardless of arbitrary
coordinate choices.

A key novel contribution is the Consensus Manifold construction: given N models, compute pairwise
CKA similarities to form a model-space distance matrix, then apply manifold embedding (MDS, UMAP)
to visualise the meta-geometry of model representation space. This reveals which architectures
are representationally equivalent, which form distinct clusters, and whether the representational
similarity correlates with test accuracy, training procedure, or architectural family.

---

## Motivation

Model distillation assumes that a teacher model's representations are worth approximating. Ensemble
methods assume that diverse models capture different aspects of the data. But neither assumption
is validated without measuring representational similarity directly. This project was built to
provide the measurement tools — making model comparison a quantitative rather than qualitative
exercise, grounded in the geometry of learned representations.

---

## Architecture

```
Set of N trained models {M₁, M₂, ..., Mₙ}
        │
  Activation extraction at matching layers
  Xᵢ ∈ R^{samples × features} per model Mᵢ
        │
  Pairwise similarity computation:
  ├── Linear CKA: k(X,Y) = ||X'Y||_F² / (||X'X||_F ||Y'Y||_F)
  ├── Kernel CKA: RBF kernel variant
  └── Representational Similarity Analysis (RSA): RDM correlation
        │
  N×N similarity matrix S
        │
  Consensus Manifold embedding (MDS / UMAP)
        │
  ├── Architecture clustering visualisation
  └── Correlation with test accuracy / training cost
```

---

## Features

### Centered Kernel Alignment (CKA)
Linear and kernel CKA implementation for measuring representational similarity between pairs of activation matrices — invariant to orthogonal transformation, isotropic scaling, and permutation of samples.

### Representational Dissimilarity Matrix (RDM)
RSA-based RDM construction and Kendall-tau correlation between RDMs from different models, providing a complementary view of representational similarity to CKA.

### Multi-Architecture Benchmark Suite
Pre-built analysis pipelines for comparing ResNet-18/50, ViT-B/16, EfficientNet-B0, and MobileNet-V3 on CIFAR-10 and ImageNet subsets.

### Consensus Manifold Visualisation
2D MDS and UMAP embedding of the N×N model similarity matrix — visualising the meta-geometry of model space and identifying representationally equivalent architectures.

### Layer-Wise CKA Heatmaps
Compute CKA between every pair of layers across two models — producing a (depth_A × depth_B) heatmap that reveals which layers in different architectures are functionally equivalent.

### Cross-Task Transfer Prediction
Measure CKA between source-task and target-task representations to predict transfer learning performance before running expensive fine-tuning experiments.

### Distillation Quality Predictor
Use CKA between teacher and student representations to measure distillation fidelity at each layer — identifying which layers are well-distilled vs. those requiring additional alignment loss.

### Consensus Score per Data Point
Per-sample consensus score: does a specific input produce similar representations across all N models? Low-consensus samples identify the most model-dependent, potentially ambiguous data points.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **PyTorch** | Model inference and hooks | Activation extraction, multi-architecture support |
| **NumPy / SciPy** | CKA computation | Matrix operations, Frobenius norms, MDS embedding |
| **UMAP-learn** | Manifold embedding | Non-linear 2D embedding of model similarity matrix |
| **torchvision** | Pre-trained models | ResNet, ViT, EfficientNet, MobileNet |
| **scikit-learn** | Similarity metrics | Kendall-tau, Spearman correlation for RDM comparison |
| **Plotly / Seaborn** | Visualisation | CKA heatmaps, consensus manifold, similarity dendrograms |
| **pandas** | Results management | CKA matrices, per-model statistics |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/Latent-Consensus-Manifold.git
cd Latent-Consensus-Manifold
python -m venv venv && source venv/bin/activate
pip install torch torchvision numpy scipy umap-learn scikit-learn plotly seaborn pandas
streamlit run app.py
```

---

## Usage

```bash
# Compare ResNet-18 and ViT-B/16 on CIFAR-10
python compare_models.py \
  --models resnet18,vit_b_16 \
  --dataset cifar10 \
  --n_samples 2000

# Full architecture benchmark
python benchmark_consensus.py \
  --models resnet18,resnet50,efficientnet_b0,vit_b_16 \
  --output consensus_matrix.csv

# Layer-wise CKA heatmap
python layer_cka.py \
  --model_a resnet18 --model_b vit_b_16 \
  --dataset cifar10 --output heatmap.png

# Visualise consensus manifold
streamlit run app.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODELS` | `resnet18,vit_b_16` | Comma-separated list of torchvision model names |
| `DATASET` | `cifar10` | Evaluation dataset: cifar10, imagenet_subset |
| `N_SAMPLES` | `2000` | Samples per class for CKA computation |
| `CKA_MODE` | `linear` | CKA variant: linear or rbf_kernel |
| `LAYER_NAMES` | `layer4,heads` | Layers to compare (comma-separated) |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
Latent-Consensus-Manifold/
├── README.md
├── requirements.txt
├── LAteNT.py
├── council.py
├── latent_dictionary.py
├── memory.py
├── Results Archive/agi_session_11290.json
├── Results Archive/agi_session_11290_latest.json
├── Results Archive/agi_session_70290.json
└── ...
```

---

## Roadmap

- [ ] LLM representation comparison: measure CKA between GPT-2, BERT, and T5 layer representations on NLP tasks
- [ ] Temporal consensus: track CKA evolution during training to visualise when representations align
- [ ] Federated learning application: use consensus scores to select optimal model aggregation weights
- [ ] Theoretical connection: empirically test whether high CKA predicts similar generalisation error bounds
- [ ] Cross-modality consensus: compare visual and textual representations from CLIP for the same concepts

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

CKA computation scales as O(N²·S²) in the naive implementation, where N is features and S is samples. For large activation matrices (>10,000 features), use the minibatch CKA estimator to avoid memory overflow. Layer-wise CKA heatmaps for deep networks can take 10–30 minutes for a full comparison.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
