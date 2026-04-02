# stroke-prediction-r

Machine learning project comparing Linear, RBF, and Nyström SVM models for stroke risk prediction using clinical data, with a focus on handling class imbalance.

## Overview

This project analyzes stroke prediction using an imbalanced healthcare dataset (~95% non-stroke, ~5% stroke). The goal is to evaluate how different SVM approaches perform under imbalance and to identify the most effective method for detecting stroke cases.

## Models

* Linear SVM
* RBF SVM
* Nyström SVM

## Approach

* Trained baseline models on original data
* Applied upsampling to handle class imbalance
* Tested SMOTE (excluded due to poor performance)

## Results

* Linear SVM achieved the highest PR-AUC (~0.22) and recall (~0.96)
* Nyström SVM provided a more balanced performance
* Upsampling improved results, while SMOTE degraded performance

## Key Insight

Handling class imbalance is critical. Simple upsampling outperformed more complex techniques like SMOTE.

## How to Run

```bash
Rscript svm_models.R
```
