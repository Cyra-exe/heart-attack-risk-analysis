# Heart Attack Risk Prediction using Edge Impulse

> ⚠️ **Disclaimer:** This project is for educational/portfolio purposes only. It uses a synthetic dataset and is not a medical device. It should not be used for real health decisions, diagnosis, or treatment.

A machine learning project that classifies heart attack risk (binary: low/high) using a neural network trained and deployed via **Edge Impulse**, targeting embedded inference on an **Espressif ESP-EYE** device.

## 📋 Overview

This project uses a tabular dataset of health and lifestyle indicators to predict heart attack risk. The model is built, trained, and evaluated entirely within Edge Impulse's Studio, then compiled with the **EON™ Compiler** for efficient on-device inference on a target microcontroller.

The project went through an iterative debugging and improvement process — from a non-functional first model to a validated, deployable classifier. That process is documented below, since diagnosing and fixing a failing model is as much a part of this project as the final result.

## 📊 Dataset

- **Type:** Synthetically generated dataset modeling realistic cardiovascular risk relationships (age, cholesterol, lifestyle factors, family history, etc.) with added noise so it is not perfectly separable — similar to real-world health data.
- **Samples:** 8,000 total, stratified 80/20 into training (6,400) and test (1,600) sets
- **Class balance:** Balanced 50/50 (4,000 / 4,000) between low-risk (0) and high-risk (1) labels

### Input Features (22 total)

**Original 18 features:**
Age, Cholesterol, Heart Rate, Diabetes, Family History, Smoking, Obesity, Alcohol Consumption, Exercise Hours Per Week, Previous Heart Problems, Medication Use, Stress Level, Sedentary Hours Per Day, Income, BMI, Triglycerides, Physical Activity Days Per Week, Sleep Hours Per Day

**Engineered interaction features (4 added):**
- `BMI × Sedentary Hours Per Day` — compounding lifestyle risk
- `Cholesterol × Triglycerides` — combined lipid panel risk
- `Age × Family History` — genetic risk compounding with age
- `Smoking × Previous Heart Problems` — compounding cardiac risk

### Output

Binary classification — **2 classes (0, 1)**, representing low vs. high heart attack risk.

## 🧠 Model Architecture

A fully-connected (dense) neural network:

```
Input layer (22 features)
   ↓
Dense layer (16 neurons)
   ↓
Dense layer (8 neurons)
   ↓
Output layer (2 classes)
```

**Training configuration:**
- Training cycles: 100
- Optimizer: Learned optimizer (enabled)
- Batch size: 64
- Validation set size: 20%
- Training processor: CPU
- Model version: Quantized (int8), also evaluated as Unoptimized (float32)

## 🔍 Development Process & Debugging

The first trained model was non-functional — it always predicted a single class (100%/0% confusion matrix, ROC AUC of 0.50, equivalent to random guessing). Diagnosing and fixing this involved several iterations:

| Stage | Change made | ROC AUC | Accuracy | Result |
|---|---|---|---|---|
| 1 | Baseline (raw, imbalanced dataset, no processing block) | 0.50 | 62.7% | Model always predicted class 0 |
| 2 | Added a normalization/processing block | 0.58 | 59.8% | Model began predicting both classes |
| 3 | Balanced dataset to 50/50 class split | 0.61 | 60.9% | Balanced confusion matrix, but performance plateaued |
| 4 | Tried larger architecture (32→16→8) | 0.59 | 60.6% | No improvement — capacity wasn't the bottleneck |
| 5 | Tried smaller batch size (32) | 0.60 | 60.0% | Confirmed hyperparameter tuning had plateaued |
| 6 | **Added 4 engineered interaction features (22 total)** | **0.74** | **73.6%** | Major improvement — this was the real unlock |

**Key finding:** Once class imbalance and missing normalization were fixed, further hyperparameter tuning (architecture size, batch size, training cycles) produced no meaningful gains — performance plateaued around 0.59–0.61 ROC AUC across multiple configurations. The breakthrough came from **feature engineering**: adding interaction terms between correlated risk factors (e.g. cholesterol × triglycerides) gave the model signal it could not otherwise construct from a shallow 2-layer network. This suggested the original bottleneck was the feature set, not model capacity.

## 📈 Final Results

### Validation Set (Quantized int8)

| Metric | Value |
|---|---|
| Accuracy | 73.6% |
| Loss | 0.50 |
| Area under ROC Curve | 0.74 |
| Weighted average Precision | 0.74 |
| Weighted average Recall | 0.74 |
| Weighted average F1 score | 0.74 |

**Confusion matrix (validation set):**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 70.4% | 29.6% |
| **Actual 1** | 23.2% | 76.8% |
| **F1 Score** | 0.73 | 0.74 |

### Held-Out Test Set (Quantized int8 — final deployment model)

| Metric | Value |
|---|---|
| Accuracy | 68.20% |
| Area under ROC Curve | 0.76 |
| Weighted average Precision | 0.76 |
| Weighted average Recall | 0.76 |
| Weighted average F1 score | 0.76 |

**Confusion matrix (test set):**

| | Predicted 0 | Predicted 1 | Uncertain |
|---|---|---|---|
| **Actual 0** | 67.1% | 17.7% | 15.3% |
| **Actual 1** | 16.1% | 69.3% | 14.6% |
| **F1 Score** | 0.73 | 0.74 | — |

ROC AUC held steady (and slightly improved) between validation (0.74) and the held-out test set (0.76), indicating the model generalizes rather than overfitting to the validation split. Quantization to int8 cost negligible performance versus the float32 version (68.02% vs 68.20% accuracy), confirming the model is deployment-ready without meaningful accuracy loss.

Roughly 15% of test samples fall into an "uncertain" bucket rather than a confident 0/1 prediction — the model declines to force a low-confidence guess rather than making an unreliable one.

### On-Device Performance (EON Compiler, Quantized int8)

| Metric | Value |
|---|---|
| Inferencing time | 2 ms |
| Peak RAM usage | 2.0K |
| Flash usage | 25.6K |
| Target device | Espressif ESP-EYE |

## ⚠️ Known Limitations

- ROC AUC of 0.76 is a solid, usable result but not a highly precise classifier — roughly a quarter of predictions are wrong or uncertain
- Trained on synthetic data with intentionally added noise; a real clinical dataset may yield different results
- Further gains would likely require additional feature engineering or a different model family (e.g. tree-based classical ML), rather than further neural network hyperparameter tuning, which was shown to plateau

## 📁 Repository Structure

```
heart-attack-risk-project/
├── README.md                                    # Project documentation (this file)
├── data/
│   ├── heart_attack_risk_dataset.csv            # Full balanced dataset (18 features)
│   ├── heart_attack_risk_train.csv              # Training split (18 features)
│   ├── heart_attack_risk_test.csv               # Test split (18 features)
│   ├── heart_attack_risk_dataset_engineered.csv # Full dataset with engineered features (22 features)
│   ├── heart_attack_risk_train_engineered.csv   # Training split (22 features, final version used)
│   └── heart_attack_risk_test_engineered.csv    # Test split (22 features, final version used)
└── assets/
    └── (screenshots: confusion matrix, feature explorer, training curves)
```

## 🚀 Model Deployment

The trained model can be exported directly from the Edge Impulse Studio project using **Deployment → Espressif ESP-EYE** (or the target of your choice: Arduino Library, C++ Library, WebAssembly, etc.).

The model export is not committed to this repository. To get the latest trained model, deploy it directly from the Edge Impulse Studio project link below.

**Project link:** [Add your Edge Impulse project URL here]

## 🛠️ Tools & Platform

- **[Edge Impulse Studio](https://edgeimpulse.com/)** — data acquisition, signal processing, model training, and deployment
- **Target hardware:** Espressif ESP-EYE
- **Python** (pandas, numpy, scikit-learn) — used to generate and split the synthetic dataset

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Author

Oluwakemi Orisanaiye (Cyra-exe) / https://github.com/Cyra-exe / www.linkedin.com/in/oluwakemi-orisanaiye-80a830355
