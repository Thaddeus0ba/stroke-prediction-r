# Stroke Risk Prediction Using Machine Learning

## Overview
This project focuses on predicting stroke risk using machine learning models.  
We compare multiple algorithms under different class imbalance handling techniques to evaluate their effectiveness in detecting rare stroke cases.

---

## Dataset
- Source: [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Samples: 5,110 patients
- Features: Demographic and clinical variables (age, BMI, glucose, etc.)
- Target: Stroke (Yes/No)
- Class Imbalance: ~4.87% stroke cases

---

## Problem
The dataset is highly imbalanced, meaning most patients do not have a stroke.  
This causes standard models to achieve high accuracy while failing to detect stroke cases.

---

## 🧠 Methodology

## 1. Data Preprocessing
- Removed ID column
- Handled missing BMI values using median imputation
- Converted categorical variables to factors
- Stratified train-test split (80/20)

---

## 2. Exploratory Data Analysis (EDA)
- Class distribution visualization
- Age distribution by stroke outcome
- Glucose level comparison

---

## 3. Models Used
- Logistic Regression
- Random Forest
- Linear SVM
- RBF SVM
- Additional kernel SVM (via `kernlab`)

---

## 4. Experimental Setup
We evaluated models under three conditions:

1. **Baseline (No imbalance handling)**
2. **Upsampling (minority class duplication)**
3. **SMOTE (synthetic data generation)**

---

##  Results

##  Baseline
- High accuracy (~95%)
- Very low recall
- Models failed to detect stroke cases

---

##  Upsampling (Best Performance)
- Recall improved significantly (~0.80–0.85)
- Models successfully detected most stroke cases
- Precision remained low (~0.12–0.14)

**Best Model:** Random Forest (highest F1-score)

---

##  SMOTE
- Improved recall in some cases
- Results were less stable compared to upsampling

---

##  Key Insights
- Accuracy is misleading in imbalanced datasets
- Recall and F1-score are more appropriate metrics
- Handling class imbalance is critical for model performance
- Upsampling provided the most consistent improvement

---

##  Limitations
- Small dataset (~5k samples)
- Low precision (high false positives)
- SMOTE may introduce unrealistic synthetic data

---

##  Future Work
- Improve precision using threshold tuning
- Explore advanced imbalance techniques
- Test additional models or feature engineering

---

##  Project Structure
scripts/
01_load_data.R
02_eda.R
03_preprocessing.R
04_models_baseline.R
04_models_upsampling.R
04_models_smote.R

outputs/
train_data.csv
test_data.csv
model_results_upsampling.csv
model_results_smote.csv

figures/
class_distribution.png
age_distribution.png
glucose_boxplot.png


---

##  Team Contributions

- **Thaddeus**: Data preprocessing, EDA, baseline models, upsampling implementation
- **Mosh**: SMOTE implementation, additional model experimentation
- **Samuel**: Report writing and interpretation

---

##  Conclusion
Machine learning models can be used to predict stroke risk, but performance depends heavily on handling class imbalance.  
Among the tested approaches, upsampling provided the most reliable improvement in detecting stroke cases.

---