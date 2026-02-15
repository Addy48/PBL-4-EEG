# Inter-Subject Variability in EEG-Based Stress Detection

**A Machine Learning Perspective**

## Overview

This project evaluates classical machine learning models for EEG-based stress classification under both subject-dependent and subject-independent validation using the SAM-40 dataset. The study quantifies the generalization ceiling of feature-engineered approaches.

**Paper Status:** Submitted to international conference (Springer LNCS). Under review.

## Dataset

- **SAM-40** (Ghosh et al., 2022)
- 40 subjects, 14 EEG channels (Emotiv EPOC+), 128 Hz
- 4 tasks: Relaxation, Arithmetic, Stroop, Mirror Image Recognition
- Binary classification: Relax vs each cognitive task

## Key Results

| Validation | Best Model | Avg Accuracy |
|---|---|---|
| Subject-Dependent | KNN | 97.39% |
| Subject-Independent | AdaBoost | 55.29% |

**Generalization Gap:** 42.1 percentage points

## Project Structure

```
├── index.html                      # Project website (GitHub Pages)
├── src/
│   ├── preprocessing.py            # EEG signal preprocessing
│   ├── feature_extraction.py       # Statistical feature extraction
│   └── eeg_classification.py       # Classification pipeline (demo)
├── results/
│   └── summary.csv                 # Aggregated results
├── requirements.txt
└── README.md
```

## Methodology

1. **Preprocessing:** Savitzky-Golay filtering, Daubechies-2 wavelet denoising
2. **Features:** Statistical (Mean, Variance, Energy, RMS, Entropy, Kurtosis, Skewness, MAD, Peak-to-Peak, Std)
3. **Models:** KNN, Random Forest, XGBoost, CatBoost, AdaBoost, Linear SVC, Logistic Regression
4. **Validation:** Subject-Dependent (StratifiedKFold) + Subject-Independent (GroupKFold)

> Full ensemble analysis and statistical validation code withheld (paper under review).

## Team

- **Prakriti Sharma** (23FE10CSE00459)
- **Aaditya Upadhyay** (23FE10CSE00457)
- **Guide:** Dr. Akshay Jadhav

School of Computer Science and Engineering, Manipal University Jaipur
