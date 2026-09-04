# Data Efficiency and Robustness Analysis: Classical ML vs. SmallResNet

An empirical evaluation investigating sample efficiency and perturbation robustness between Classical Machine Learning pipelines (PCA + SVM / Random Forest) and a Convolutional Neural Network (`SmallResNet`) on a balanced CIFAR-10 subset.

---

## 📌 Executive Summary

This study establishes formal falsification criteria to test two foundational hypotheses regarding deep learning versus feature-based models:

* **Hypothesis 1 (Data Size Scaling):** **Rejected.** Classical ML significantly outperforms SmallResNet in low-data regimes ($\le 10\%$ data) by $+7.03\%$ and $+10.27\%$ ($t = 10.65, p = 0.0087$), violating the pre-experimental $\le 5\%$ equivalence boundary. CNN only achieves statistical superiority at $100\%$ scale ($+7.71\%, p = 0.018$).
* **Hypothesis 2 (Noise Robustness & Error Propagation):** **Rejected.** The assumption that multi-layer convolutional networks amplify noise faster than shallow models is disproven. Under mild Gaussian perturbations ($\sigma = 0 \to 12.75$), Classical ML immediately collapses by $30.78\%$, while SmallResNet only drops by $5.53\%$, proving over 5x greater early-stage noise resilience.

---

## 📂 Repository Structure

```text
├── data/                               # Raw dataset directory (ignored by git)
├── Results/                            # Visual artifacts and generated plots
│   ├── Learning_Curve_Mean.png         # Scaling curve (mean accuracy across seeds)
│   ├── Learning_Curve_Max.png          # Scaling curve (peak accuracy)
│   └── Robustness_Curve.png            # Degradation curve across noise levels
├── cifar10_subset.npz                  # Processed stratified CIFAR-10 subset
├── BuildSubset.py                      # Subset extraction and preprocessing
├── PipelineML.py                       # Classical ML training and evaluation pipeline
├── PipelineResNet.py                   # PyTorch SmallResNet architecture & training loop
├── Sweep.py                            # Grid search and hyperparameter sweeping
├── Log Processing.py                   # Raw log aggregation and statistics extraction
├── Hypothesis Verification.py          # Paired t-tests, drop rate, and hypothesis checks
├── Schema Log.xlsx                     # Structured experiment log
├── Statistical_Test_Results.xlsx       # Comprehensive hypothesis testing artifacts
├── Hypothesis Verification Report.docx  # Final comprehensive technical report
└── Hypothesis Verification Report.pdf   # Exported publication-ready document
```

---

## 🔬 Experimental Methodology & Findings

### 1. Data Scaling Verification (H1)
* Models were evaluated across fractions: $5\%, 10\%, 25\%, 50\%, 100\%$ of training data.
* Classical feature extractors demonstrated superior inductive biases under data starvation ($\le 10\%$), whereas CNN requires a minimum sample critical mass ($\approx 50\%$) to break even and eventually outperform.

| Percentage Data | ML Mean Acc (%) | DL Mean Acc (%) | Difference (ML - DL) | $p$-value | H1 Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5% | 34.80 | 27.77 | +7.03 | 0.092 | **VIOLATED** |
| 10% | 39.14 | 28.87 | +10.27 | 0.009 | **VIOLATED** |
| 25% | 42.75 | 39.70 | +3.05 | 0.064 | Satisfied |
| 50% | 44.06 | 45.20 | -1.14 | 0.051 | Satisfied |
| 100% | 45.36 | 53.07 | -7.71 | 0.018 | Satisfied |

### 2. Noise Degradation & Robustness Paradox (H2)
* Tested under additive zero-mean Gaussian noise $\sigma \in \{0.0, 12.75, 38.25, 76.50\}$.
* While CNN displays a higher overall linear degradation slope ($0.5142$ vs. $0.4508$), interval analysis confirms this is an artifact of CNN's significantly higher ceiling baseline ($53.07\%$ vs. $45.36\%$). Classical ML degrades steeply in the first noise band and reaches near-chance performance early.

---

## 🚀 Reproduction & Usage

### 1. Environment Setup
```bash
git clone https://github.com/khoa-td/Data-Efficiency-and-Robustness-Classical-ML-vs.-CNN-ResNet-.git
cd Data-Efficiency-and-Robustness-Classical-ML-vs.-CNN-ResNet-
pip install numpy torch torchvision scikit-learn scipy pandas openpyxl matplotlib
```

### 2. Run Pipeline & Hypothesis Verification
```bash
# 1. Build CIFAR-10 stratified subset
python BuildSubset.py

# 2. Run scaling and noise sweeps
python Sweep.py

# 3. Aggregate log schemas and compute paired t-tests
python "Log Processing.py"
python "Hypothesis Verification.py"
```

---

## 📄 Artifacts
The full breakdown of tests, methodology derivations, and analytical discussions are accessible in [Hypothesis Verification Report.pdf](./Hypothesis%20Verification%20Report.pdf).