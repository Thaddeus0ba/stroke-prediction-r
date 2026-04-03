library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(pROC)
library(kernlab)
library(themis)
library(recipes)

# 1. Load data
train_data <- read_csv("outputs/train_data.csv", show_col_types = FALSE)
test_data  <- read_csv("outputs/test_data.csv", show_col_types = FALSE)

# 2. Fix labels
train_data$stroke <- ifelse(train_data$stroke == "Stroke", "Stroke", "No_Stroke")
test_data$stroke  <- ifelse(test_data$stroke == "Stroke", "Stroke", "No_Stroke")

train_data$stroke <- factor(train_data$stroke, levels = c("Stroke", "No_Stroke"))
test_data$stroke  <- factor(test_data$stroke, levels = c("Stroke", "No_Stroke"))

# 3. Convert categorical columns
factor_cols <- c(
  "gender", "ever_married", "work_type",
  "Residence_type", "smoking_status",
  "hypertension", "heart_disease"
)

train_data[factor_cols] <- lapply(train_data[factor_cols], factor)
test_data[factor_cols]  <- lapply(test_data[factor_cols], factor)

# SMOTE (TRAIN ONLY)
rec <- recipe(stroke ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(stroke)

# TRAIN the recipe
rec_prep <- prep(rec, training = train_data)

# APPLY to train
train_smote <- bake(rec_prep, new_data = NULL)

# APPLY to test
test_processed <- bake(rec_prep, new_data = test_data)

# 5. Train control
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# 6. Logistic Regression
set.seed(42)
model_log <- train(
  stroke ~ .,
  data = train_smote,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

# 7. Random Forest
set.seed(42)
model_rf <- train(
  stroke ~ .,
  data = train_smote,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  ntree = 200
)

# 8. Linear SVM
set.seed(42)
model_svm_linear <- train(
  stroke ~ .,
  data = train_smote,
  method = "svmLinear",
  trControl = ctrl,
  metric = "ROC"
)

# 9. RBF SVM
set.seed(42)
model_svm_rbf <- train(
  stroke ~ .,
  data = train_smote,
  method = "svmRadial",
  trControl = ctrl,
  metric = "ROC"
)

# 10. Nyström SVM (kernlab)
set.seed(42)
model_nystrom <- ksvm(
  stroke ~ .,
  data = train_smote,
  kernel = "rbfdot",
  prob.model = TRUE
)

# 11. Prediction helper
predict_model <- function(model, name, is_kernlab = FALSE) {
  
  if (is_kernlab) {
    probs <- predict(model, test_processed, type = "probabilities")[, "Stroke"]
    preds <- ifelse(probs > 0.5, "Stroke", "No_Stroke")
    preds <- factor(preds, levels = c("Stroke", "No_Stroke"))
  } else {
    preds <- predict(model, test_processed)
    probs <- predict(model, test_processed, type = "prob")[, "Stroke"]
  }

  cm <- confusionMatrix(preds, test_data$stroke)
  roc_obj <- roc(response = test_data$stroke, predictor = probs)

  list(
    name = name,
    confusion = cm,
    AUC = auc(roc_obj)
  )
}

# 12. Collect results
results <- list(
  predict_model(model_log, "Logistic Regression"),
  predict_model(model_rf, "Random Forest"),
  predict_model(model_svm_linear, "Linear SVM"),
  predict_model(model_svm_rbf, "RBF SVM"),
  predict_model(model_nystrom, "Nyström SVM", TRUE)
)

# 13. Print results
for (res in results) {
  cat("\n=========================\n")
  cat("Model:", res$name, "\n")
  print(res$confusion)
  cat("AUC:", as.numeric(res$AUC), "\n")
}