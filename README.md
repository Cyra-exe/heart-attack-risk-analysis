# Heart Attack Risk Prediction using Edge Impulse

> ⚠️ **Disclaimer:** This project is for educational/portfolio purposes only. It is not a medical device and should not be used for real health decisions, diagnosis, or treatment.

A machine learning project that classifies heart attack risk (binary: low/high) using a neural network trained and deployed via **Edge Impulse**, targeting embedded inference on an **Espressif ESP-EYE** device.

## 📋 Overview

This project uses a tabular dataset of health and lifestyle indicators to predict heart attack risk. The model is built, trained, and evaluated entirely within Edge Impulse's Studio, then compiled with the **EON™ Compiler** for efficient on-device inference on a target microcontroller.

## 📊 Dataset Source

- **Name:** Heart Attack Risk Prediction Dataset.
- **Source:** https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset
- **License:** CC BY 4.0
- **Description:** This synthetic dataset provides a comprehensive array of features relevant to heart health and lifestyle choices, encompassing patient-specific details and it is a synthetic creation generated using ChatGPT to simulate a realistic experience.

## 📈 Dataset Summary

| Detail | Value |
|---|---|
| Training samples | 5,607 |
| Test samples | 1,402 |
| Split | 80% / 20% |

### Input Features (18)

- Age
- Cholesterol
- Heart Rate
- Diabetes
- Family History
- Smoking
- Obesity
- Alcohol Consumption
- Exercise Hours Per Week
- Previous Heart Problems
- Medication Use
- Stress Level
- Sedentary Hours Per Day
- Income
- BMI
- Triglycerides
- Physical Activity Days Per Week
- Sleep Hours Per Day

### Output

Binary classification — **2 classes (0, 1)**, representing low vs. high heart attack risk.

## 🧠 Model Architecture

A fully-connected (dense) neural network:

```
Input layer (18 features)
   ↓
Dense layer (30 neurons)
   ↓
Dense layer (50 neurons)
   ↓
Dense layer (70 neurons)
   ↓
Dense layer (50 neurons)
   ↓
Dense layer (30 neurons)
   ↓
Output layer (2 classes)
```

**Training configuration:**
- Training cycles: 30
- Optimizer: Learned optimizer (enabled)
- Training processor: CPU
- Model version: Quantized (int8)

## 📈 Results

### Performance (Validation Set)

| Metric | Value |
|---|---|
| Accuracy | 62.7% |
| Loss | 0.66 |
| Area under ROC Curve | 0.50 |
| Weighted average Precision | 0.39 |
| Weighted average Recall | 0.63 |
| Weighted average F1 score | 0.48 |

### Confusion Matrix (Validation Set)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 100% | 0% |
| **Actual 1** | 100% | 0% |
| **F1 Score** | 0.77 | 0.00 |

### On-Device Performance (EON Compiler, Quantized int8)

| Metric | Value |
|---|---|
| Inferencing time | 2 ms |
| Peak RAM usage | 2.0K |
| Flash usage | 25.6K |
| Target device | Espressif ESP-EYE |

## ⚠️ Known Limitations

The current model shows signs of underfitting and class imbalance:
- ROC AUC of ~0.50 suggests performance close to random guessing
- The confusion matrix indicates the model is not correctly identifying class 1 (high-risk) cases
- F1 score for class 1 is 0.00

**Planned improvements:**
- Address class imbalance (e.g. oversampling, class weighting)
- Add a feature scaling/normalization processing block
- Architecture and hyperparameter tuning
- Increase training cycles and monitor for overfitting/underfitting

## 📁 Repository Structure

```
heart-attack-risk-project/
└── README.md          # Project documentation (this file)
```

*(Structure will be updated as scripts, notebooks, and exported model files are added.)*

## 🚀 Model Deployment

The trained model can be exported directly from the [Edge Impulse Studio project link — fill in] using **Deployment → [target: e.g. ESP-EYE / Arduino Library / C++ Library]**.

The model export is not committed to this repository. To get the latest trained model, deploy it directly from the Edge Impulse Studio project.

## 🛠️ Tools & Platform

- **[Edge Impulse Studio](https://edgeimpulse.com/)** — data acquisition, signal processing, model training, and deployment
- **Target hardware:** Espressif ESP-EYE

## 📄 License

MIT License.

## 🙋 Author

Oluwakemi Orisanaiye / https://github.com/Cyra-exe / linkedin.com/in/oluwakemi-orisanaiye-80a830355
