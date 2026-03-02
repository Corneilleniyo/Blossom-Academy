# Industrial Anomaly Detection (Generator Fuel & Runtime) — Machine Learning Project

## Project Overview
This project builds a multi-class anomaly detection system for industrial generator operational data (runtime and fuel-consumption signals). The objective is to detect abnormal patterns and support reliability monitoring and decision-making.

## Problem Statement
Real-world operational logs typically contain:
- severe class imbalance (rare anomaly classes)
- noisy measurements and missing values
- cluster/site-specific distribution shifts
- costly false negatives (missed anomalies)

Because of imbalance, accuracy can be misleading; we prioritize metrics that reflect minority-class performance.

## Methodology
1. **Data Preparation**
   - cleaning, missing-value handling, outlier checks  
   - feature engineering (usage intensity, fuel-rate proxies, temporal aggregates)

2. **Model Development**
   - baseline models and a supervised multi-class classifier
   - imbalance handling using resampling (e.g., SMOTE-ENN where appropriate)

3. **Evaluation**
   - Macro-F1 (balanced view across classes)
   - ROC-AUC (multi-class, where applicable)
   - Recall for minority anomaly classes (reduce missed anomalies)

4. **Explainability**
   - SHAP-based feature attribution to justify anomaly decisions and support stakeholders

## Key Outcomes
- improved detection of minority anomaly classes compared to naive baselines
- more reliable evaluation using macro-averaged metrics
- interpretable insights into drivers of anomaly predictions

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, SHAP

## Next Improvements
- domain adaptation for cluster/site shift
- stricter label validation and error analysis
- lightweight deployment pipeline for production monitoring

## Author
Corneille Niyonkuru — MSc Data Science (AIMS Rwanda), BSc Mechanical Engineering (UR-CST)
