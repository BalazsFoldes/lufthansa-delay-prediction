# Aviation Delay Prediction
### Lufthansa Systems — Challenge

Binary classification pipeline predicting whether a flight will be delayed by more than 15 minutes.

---

## Project Structure

```
lufthansa-delay-prediction/
│
├── data/
│   └── aviation_delay.csv          # Raw dataset (6,000 flights, Jan–May 2024)
│
├── notebooks/
│   └── analysis.ipynb              # Full analysis walkthrough (EDA → modelling → results)
│
├── outputs/
│   ├── figures/                    # All generated plots (EDA, ROC curves, confusion matrices, feature importances)
│   └── results/
│       └── model_comparison.csv    # Final model comparison table
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths, constants, feature lists, leakage columns
│   ├── data_loader.py              # Data loading, leakage reporting, time-based split
│   ├── eda.py                      # Exploratory visualisations
│   ├── evaluation.py               # Threshold calibration, metrics, plots
│   ├── models.py                   # Model registry and training loop
│   └── preprocessing.py            # Imputation, encoding, scaling
│
├── DOKUMENTACIO.md                 # Full documentation in hungarian
├── README.md
├── requirements.txt
└── run_pipeline.py                 # End-to-end pipeline entry point
```

---

## Quickstart

### 1. Clone & set up environment

```bash
git clone https://github.com/your-username/lufthansa-delay-prediction.git
cd lufthansa-delay-prediction

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Add the dataset

Place `aviation_delay.csv` in the `data/` directory.

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

This will:
- Load and validate the raw data
- Report and remove leakage columns
- Apply time-based train / val / test split
- Preprocess features (imputation, encoding, scaling)
- Train 6 models (Dummy → LightGBM)
- Calibrate decision thresholds on the validation set
- Evaluate all models on the held-out test set
- Save results to `outputs/results/` and plots to `outputs/figures/`

### 4. Run the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Methodology

### Data Leakage
Five columns were identified and removed before any modelling — they are only available *after* a flight has departed or landed:

| Column | Reason |
|---|---|
| `actual_delay_minutes` | Target is derived directly from this |
| `actual_gate_out_time_diff` | Measured after gate-out |
| `maintenance_closed_after_pushback` | Post-pushback event |
| `final_delay_reason` | Assigned after delay occurs |
| `sched_buffer_mins_latest` | Identical to `turnaround_minutes` |

### Validation Strategy
Time-based split — no data from the future is ever used during training:

```
Jan ──── Feb ──── Mar ──── Apr ──── May
[       TRAIN (4,368)      ] [ VAL ] [TEST]
```

### Threshold Calibration
The 0.5 default decision threshold is too conservative at a 96/4 class ratio. For each model, the F1-maximising threshold is found on the **validation set** and then applied — without adjustment — to the **test set**.

---

## Results

| Model | F1 | ROC-AUC | Threshold |
|---|---|---|---|
| Dummy (majority) | 0.000 | 0.500 | 0.50 |
| **Logistic Regression** | **0.200** | **0.748** | 0.35 |
| Decision Tree | 0.121 | 0.565 | 0.25 |
| Random Forest | 0.000 | 0.710 | 0.40 |
| XGBoost | 0.077 | 0.654 | 0.20 |
| LightGBM | 0.050 | 0.673 | 0.15 |

**Logistic Regression** achieves the best F1 (0.200) and ROC-AUC (0.748), capturing 75% of all real delays on the test set. Ensemble models show strong ranking ability (ROC-AUC 0.65–0.71) but their val-set thresholds do not generalise to the 8-positive-case test set — an expected consequence of the synthetic, largely random data.

> Primary metrics are **F1** and **ROC-AUC**. Accuracy is misleading at 96/4 class balance — a model that always predicts "no delay" scores 95.8% accuracy while catching zero delays.

---

## Documentation

Full analysis documentation (in Hungarian) is available in [`DOKUMENTACIO.md`](DOKUMENTACIO.md), covering:

- EDA findings and data quality issues
- Leakage analysis and justification
- Modelling decisions and validation strategy
- Business interpretation and production deployment recommendations

---

## Requirements

```
python >= 3.9
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0
lightgbm >= 4.0.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
jupyter >= 1.0.0
```

---

## Reproducibility

Every result in this project is fully reproducible with a single command:

```bash
python run_pipeline.py
```

Random seeds are fixed via `RANDOM_STATE = 42` in `src/config.py`.