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
# 4. Train control
# -----------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# -----------------------------
# 5. Logistic Regression
# -----------------------------
set.seed(42)
model_log <- train(
  stroke ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

# -----------------------------
# 6. Random Forest
# -----------------------------
set.seed(42)
model_rf <- train(
  stroke ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  ntree = 200
)

# -----------------------------
# 7. Linear SVM
# -----------------------------
set.seed(42)
model_svm_linear <- train(
  stroke ~ .,
  data = train_data,
  method = "svmLinear",
  trControl = ctrl,
  metric = "ROC"
)

# -----------------------------
# 8. RBF SVM
# -----------------------------
set.seed(42)
model_svm_rbf <- train(
  stroke ~ .,
  data = train_data,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC"
)

# -----------------------------
# 9. Prediction helper
# -----------------------------
predict_model <- function(model, name) {
  preds <- predict(model, test_data)
  probs <- predict(model, test_data, type = "prob")[, "Stroke"]
  
  cm <- confusionMatrix(preds, test_data$stroke)
  roc_obj <- roc(response = test_data$stroke, predictor = probs)
  
  list(
    name = name,
    confusion = cm,
    AUC = auc(roc_obj)
  )
}

# -----------------------------
# 10. Collect results
# -----------------------------
results <- list(
  predict_model(model_log, "Logistic Regression"),
  predict_model(model_rf, "Random Forest"),
  predict_model(model_svm_linear, "Linear SVM"),
  predict_model(model_svm_rbf, "RBF SVM")
)

# -----------------------------
# 11. Print results
# -----------------------------
for (res in results) {
  cat("\n=========================\n")
  cat("Model:", res$name, "\n")
  print(res$confusion)
  cat("AUC:", as.numeric(res$AUC), "\n")
}