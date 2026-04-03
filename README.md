# Stroke Prediction (R)

## Overview
This project builds machine learning models to predict stroke risk using healthcare data.

The goal is to compare multiple models and evaluate their ability to detect stroke cases.

---

## Models Used
- Logistic Regression  
- Random Forest  
- Linear SVM  
- RBF SVM  
- Nyström SVM (kernel approximation)  

---

## Key Finding
All baseline models achieved high accuracy (~95%) but extremely low recall (~0%).

This means:
- Models correctly predict **No Stroke**
- But fail to detect actual **Stroke cases**

This is due to severe class imbalance in the dataset.

---

## Project Structure


stroke-prediction-r/
│
├── data/
│ └── healthcare-dataset-stroke-data.csv
│
├── figures/
│ ├── age_distribution.png
│ ├── class_distribution.png
│ └── glucose_boxplot.png
│
├── outputs/
│ ├── train_data.csv
│ ├── test_data.csv
│ ├── model_results.csv
│ ├── model_results_smote.csv
│ └── model_results_upsampling.csv
│
├── scripts/
│ ├── 01_load_data.R
│ ├── 02_eda.R
│ ├── 03_preprocessing.R
│ ├── 04_models_baseline.R
│ ├── 04_models_smote.R
│ └── 04_models_upsampling.R
│
├── README.md
└── stroke-prediction-r.Rproj


---

## How to Run

Run scripts in order:

```bash
Rscript scripts/01_load_data.R
Rscript scripts/02_eda.R
Rscript scripts/03_preprocessing.R
Rscript scripts/04_models_baseline.R

Optional (handle imbalance):

Rscript scripts/04_models_smote.R
Rscript scripts/04_models_upsampling.R
Outputs

Saved in:

outputs/

Includes:

Model performance metrics (Accuracy, Recall, Precision, F1 Score, AUC)
Processed training and test datasets
Nyström SVM

A Nyström approximation is used to simulate an RBF kernel efficiently.

Steps:

Sample landmark points
Compute RBF kernel matrix
Perform eigen decomposition
Transform features
Train linear SVM on transformed data
Next Steps
Improve recall using SMOTE and upsampling
Tune model hyperparameters
Explore additional models
Focus on detecting stroke cases rather than overall accuracy
Libraries Used
tidyverse
caret
e1071
randomForest
pROC
Summary

Baseline models are not sufficient due to class imbalance.

Future work will focus on improving the detection of stroke cases rather than maximizing accuracy.

Then push:
git add README.md
git commit -m "Clean README (remove emojis and fix formatting)"
git push