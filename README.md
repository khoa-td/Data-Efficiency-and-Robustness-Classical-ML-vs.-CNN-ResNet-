# Data Efficiency and Robustness: Classical ML vs. SmallResNet

An empirical study evaluating sample efficiency and Gaussian noise robustness between Classical ML pipelines (HOG + PCA + SVM/RF/LR) and a CNN (`SmallResNet`) on a stratified CIFAR-10 subset.

---

## Research Hypotheses & Verdicts

* **Hypothesis 1 (Data Size Scaling):** Assumed both models achieve similar performance under limited data ($\le 10\%$), with the CNN outperforming when training data exceeds $30\%$.  
  **Status: Rejected.** Classical ML significantly outperforms CNN at $5\%$ ($+7.03\%$) and $10\%$ data ($+10.27\%, p = 0.0087$). CNN only achieves a statistically significant advantage at $100\%$ scale ($+7.71\%, p = 0.0180$).
* **Hypothesis 2 (Noise Robustness):** Assumed CNN suffers faster accuracy degradation due to multi-layer error propagation under increasing noise ($\sigma \in \{0, 12.75, 38.25, 76.5\}$).  
  **Status: Rejected.** Under initial mild noise ($\sigma = 12.75$), Classical ML accuracy collapses by $30.78\%$, whereas CNN degrades by only $5.53\%$. CNN demonstrates superior early-stage resilience.

---

## Key Results

| Data Fraction | Classical ML (%) | CNN (%) | Difference (ML - DL) | $p$-value | Significance ($p < 0.05$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5%   | 34.80 | 27.77 | +7.03% | 0.0920 | No  |
| 10%  | 39.14 | 28.87 | +10.27%| 0.0087 | Yes |
| 25%  | 42.75 | 39.70 | +3.05% | 0.0640 | No  |
| 50%  | 44.06 | 45.20 | -1.14% | 0.0512 | No  |
| 100% | 45.36 | 53.07 | -7.71% | 0.0180 | Yes |

---

## Repository Structure

```text
├── data/
│   └── cifar10_subset.npz
├── docs/
│   ├── Research.pdf
│   └── Hypothesis_Verification_Report.pdf
├── results/
│   ├── figures/
│   └── schema_log.xlsx
├── src/
│   ├── build_subset.py
│   ├── pipeline_ml.py
│   ├── pipeline_resnet.py
│   ├── sweep.py
│   ├── log_processing.py
│   └── hypothesis_verification.py
├── .gitignore
└── README.md

## Reproduction & Usage

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

## Artifacts
The full breakdown of tests, methodology derivations, and analytical discussions are accessible in [Hypothesis Verification Report.pdf](./Hypothesis%20Verification%20Report.pdf).