# File Index - Riemannian Manifold GAT Fraud Detection

## English Files (All comments and documentation in English)

### Main Implementation
- **`fraud_detection_riemannian_gat_en.py`** (31KB)
  - Complete implementation with Riemannian manifolds + GAT
  - All comments, docstrings, and print statements in English
  - Three manifolds: Hyperbolic, SPD, Sphere
  - Geodesic optimization and pullback operations
  - ROC-AUC: 0.9965

### Documentation
- **`RIEMANNIAN_GAT_DOCUMENTATION.md`** (9.9KB)
  - Comprehensive technical documentation
  - Mathematical formulations and proofs
  - Architecture diagrams and explanations
  - Implementation details and best practices
  - Performance analysis and comparisons

- **`README_EN.md`** (7.4KB)
  - Quick start guide
  - Usage examples
  - Installation instructions
  - Troubleshooting tips
  - Citation information

## Japanese Files (日本語のファイル)

### Implementations
- **`fraud_detection_riemannian_gat.py`** (31KB)
  - リーマン多様体 + GAT の完全実装（日本語コメント）
  
- **`fraud_detection_gat.py`** (18KB)
  - GAT ベースの実装（日本語コメント）
  
- **`fraud_detection_jax.py`** (14KB)
  - VAE ベースの実装（日本語コメント）

### Documentation (Japanese)
- **`riemannian_gat_documentation.md`** (8.7KB)
  - リーマン多様体GAT の詳細ドキュメント
  
- **`gat_model_documentation.md`** (5.9KB)
  - GAT モデルの詳細説明
  
- **`model_documentation.md`** (4.1KB)
  - VAE モデルの詳細説明

## Visualizations

### Result Plots
- **`fraud_detection_riemannian_gat_results.png`** (819KB)
  - 6-panel visualization including:
    - Training loss curves
    - Anomaly score distributions
    - ROC curve (AUC = 0.9965)
    - Time series anomaly scores
    - Hyperbolic space projection (Poincaré disk)
    - Sphere manifold projection

- **`fraud_detection_gat_results.png`** (546KB)
  - 4-panel GAT model results
  - ROC-AUC: 0.9961

- **`fraud_detection_results.png`** (433KB)
  - 4-panel VAE model results
  - ROC-AUC: 0.9920

## File Relationships

```
English Version (Recommended for International Use)
├── fraud_detection_riemannian_gat_en.py  ← Main implementation
├── RIEMANNIAN_GAT_DOCUMENTATION.md       ← Detailed docs
└── README_EN.md                          ← Quick start

Japanese Version (日本語版)
├── fraud_detection_riemannian_gat.py     ← メイン実装
├── riemannian_gat_documentation.md       ← 詳細ドキュメント
├── fraud_detection_gat.py                ← GAT版
├── gat_model_documentation.md            ← GATドキュメント
├── fraud_detection_jax.py                ← VAE版
└── model_documentation.md                ← VAEドキュメント
```

## Model Comparison

| Model | File | ROC-AUC | Precision | Features |
|-------|------|---------|-----------|----------|
| VAE | fraud_detection_jax.py | 0.9920 | 74% | Latent space learning |
| GAT | fraud_detection_gat.py | 0.9961 | 97% | Graph structure learning |
| **Riemannian GAT** | fraud_detection_riemannian_gat_en.py | **0.9965** | **97%** | **Geometric learning** |

## Key Features

### Riemannian Manifold Implementation
1. **Hyperbolic Space (Poincaré Ball)**
   - Models hierarchical transaction frequency
   - Exponential and logarithmic maps
   - Möbius addition for operations

2. **SPD Manifold**
   - Covariance structure representation
   - Matrix exponentials and logarithms
   - Geometric mean (Karcher mean)

3. **Sphere Manifold**
   - Normalized directional features
   - Geodesic distances
   - Spherical projections

### Geodesic Optimization
- Karcher mean computation
- Manifold-aware feature refinement
- Geometric regularization term

### Graph Attention Network
- Multi-head attention (4 heads)
- k-NN graph construction (k=10)
- Learnable attention weights

## Running the Code

### English Version
```bash
python fraud_detection_riemannian_gat_en.py
```

### Japanese Version
```bash
python fraud_detection_riemannian_gat.py
```

Both produce identical results but with different language output.

## Dependencies

```bash
pip install jax jaxlib flax optax scikit-learn matplotlib numpy
```

## Output Files Generated

When you run the scripts, they generate:
- Training loss plots
- ROC curves
- Anomaly score distributions
- Manifold visualizations
- Confusion matrices
- Classification reports

## Performance Metrics

### Riemannian GAT (Best Performance)
- **ROC-AUC**: 0.9965
- **Precision**: 97%
- **Recall**: 43%
- **Accuracy**: 94%
- **F1-Score**: 0.60 (fraud class)

### Why Highest Performance?
1. Multi-scale geometric representations
2. Manifold-specific optimization
3. Graph structure + geometry
4. Geodesic regularization

## File Sizes

| Category | Total Size |
|----------|------------|
| Python Scripts | ~126 KB |
| Documentation | ~37 KB |
| Visualizations | ~1.8 MB |
| **Total** | **~2.0 MB** |

## Recommended Starting Point

**For English speakers**: Start with `README_EN.md` → `fraud_detection_riemannian_gat_en.py` → `RIEMANNIAN_GAT_DOCUMENTATION.md`

**For Japanese speakers**: Start with `riemannian_gat_documentation.md` → `fraud_detection_riemannian_gat.py`

## Contact & Support

For questions about the implementation:
1. Check the documentation files
2. Review the inline comments in the code
3. Examine the mathematical formulations in the docs

---

**Note**: The English version (`fraud_detection_riemannian_gat_en.py`) is recommended for publication, sharing, and international collaboration. All variable names, function names, and code structure are identical between English and Japanese versions—only comments and documentation differ.
