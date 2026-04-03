library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(pROC)

# -----------------------------
# 1. Load preprocessed data
# -----------------------------
train_data <- read_csv("outputs/train_data.csv", show_col_types = FALSE)
test_data  <- read_csv("outputs/test_data.csv", show_col_types = FALSE)

# -----------------------------
# 2. Fix target labels for caret
# caret needs valid R variable names for class probabilities
# -----------------------------
train_data$stroke <- ifelse(train_data$stroke == "Stroke", "Stroke", "No_Stroke")
test_data$stroke  <- ifelse(test_data$stroke == "Stroke", "Stroke", "No_Stroke")

train_data$stroke <- factor(train_data$stroke, levels = c("Stroke", "No_Stroke"))
test_data$stroke  <- factor(test_data$stroke, levels = c("Stroke", "No_Stroke"))

# -----------------------------
# 3. Convert categorical predictors back to factors
# -----------------------------
factor_cols <- c(
  "gender", "ever_married", "work_type",
  "Residence_type", "smoking_status",
  "hypertension", "heart_disease"
)

train_data[factor_cols] <- lapply(train_data[factor_cols], factor)
test_data[factor_cols]  <- lapply(test_data[factor_cols], factor)

# -----------------------------
# 4. Train control with UPSAMPLING
# Upsampling is applied only within the training folds
# -----------------------------
ctrl_up <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  sampling = "up"
)

# -----------------------------
# 5. Logistic Regression
# -----------------------------
set.seed(42)
model_log_up <- train(
  stroke ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl_up,
  metric = "ROC"
)

# -----------------------------
# 6. Random Forest
# -----------------------------
set.seed(42)
model_rf_up <- train(
  stroke ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl_up,
  metric = "ROC",
  ntree = 200
)

# -----------------------------
# 7. Linear SVM
# -----------------------------
set.seed(42)
model_svm_linear_up <- train(
  stroke ~ .,
  data = train_data,
  method = "svmLinear",
  trControl = ctrl_up,
  metric = "ROC"
)

# -----------------------------
# 8. RBF SVM
# -----------------------------
set.seed(42)
model_svm_rbf_up <- train(
  stroke ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = ctrl_up,
  metric = "ROC"
)

# -----------------------------
# 9. Helper function to evaluate models
# -----------------------------
predict_model <- function(model, name) {
  preds <- predict(model, test_data)
  probs <- predict(model, test_data, type = "prob")[, "Stroke"]
  
  cm <- confusionMatrix(preds, test_data$stroke)
  roc_obj <- roc(response = test_data$stroke, predictor = probs)
  
  tibble(
    Model = name,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Kappa = as.numeric(cm$overall["Kappa"]),
    Recall = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Precision = as.numeric(cm$byClass["Pos Pred Value"]),
    F1 = as.numeric(cm$byClass["F1"]),
    AUC = as.numeric(auc(roc_obj))
  )
}

# -----------------------------
# 10. Collect model results
# -----------------------------
results_table_up <- bind_rows(
  predict_model(model_log_up, "Logistic Regression"),
  predict_model(model_rf_up, "Random Forest"),
  predict_model(model_svm_linear_up, "Linear SVM"),
  predict_model(model_svm_rbf_up, "RBF SVM")
)

# -----------------------------
# 11. Print results table
# -----------------------------
print(results_table_up)

# -----------------------------
# 12. Save results
# -----------------------------
write_csv(results_table_up, "outputs/model_results_upsampling.csv")