# Bank Customer Churn Prediction Using Machine Learning

## Project Overview

This project develops a machine learning pipeline to predict customer churn in a retail banking context. The objective is to identify customers likely to leave the bank, enabling proactive retention strategies.

## Problem Statement

Customer churn significantly impacts revenue in financial institutions. The challenge lies in:

- Class imbalance between churned and non-churned customers  
- Correlated financial and demographic features  
- Trade-off between precision and recall  
- Interpretability of model predictions  

A naive accuracy-based evaluation is insufficient due to imbalance.

---

## Dataset Description

The dataset includes customer-level information such as:

- Credit score  
- Age  
- Balance  
- Estimated salary  
- Tenure  
- Number of products  
- Geography  
- Gender  
- Active membership status  

Target variable:
- `Exited` (1 = churned, 0 = retained)

---

## Methodology

### 1. Data Preprocessing
- Handling categorical variables (encoding)
- Feature scaling where necessary
- Train-test split
- Addressing class imbalance

### 2. Model Development
Models explored:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Support Vector Machine  

Hyperparameter tuning was performed to optimize performance.

---

## Evaluation Metrics

Due to class imbalance, evaluation focused on:

- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

Special emphasis was placed on **Recall**, since missing a churned customer (false negative) has higher business cost.

---

## Key Results

- Gradient Boosting achieved the best overall performance.
- Recall improved compared to baseline logistic regression.
- ROC-AUC demonstrated strong class separability.
- Feature importance analysis identified age, balance, and active membership as strong predictors.

---

## Model Interpretation

Feature importance analysis was conducted to understand key churn drivers.

Important factors included:

- Age
- Account balance
- Activity status
- Number of products

Interpretability was considered critical for financial decision-making contexts.

---

## Deployment

The trained pipeline was exported using `joblib` and integrated into a simple Streamlit application for real-time prediction.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Joblib

---

## Future Improvements

- Handling imbalance using SMOTE or cost-sensitive learning
- SHAP explainability integration
- Cross-validation for more robust performance estimates
- Fairness analysis across demographic groups

---

## Author

Corneille Niyonkuru  
MSc Mathematical Sciences (Data Science)  
African Institute for Mathematical Sciences (AIMS Rwanda)  
