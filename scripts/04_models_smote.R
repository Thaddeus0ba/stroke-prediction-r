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

# -----------------------------
# SMOTE (TRAIN ONLY)
# -----------------------------
rec <- recipe(stroke ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(stroke)

rec_prep <- prep(rec, training = train_data)

train_smote <- bake(rec_prep, new_data = NULL)
test_processed <- bake(rec_prep, new_data = test_data)

# -----------------------------
# Train control
# -----------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# -----------------------------
# Models
# -----------------------------
set.seed(42)
model_log <- train(stroke ~ ., data = train_smote, method = "glm",
                   family = "binomial", trControl = ctrl, metric = "ROC")

set.seed(42)
model_rf <- train(stroke ~ ., data = train_smote, method = "rf",
                  trControl = ctrl, metric = "ROC", ntree = 200)

set.seed(42)
model_svm_linear <- train(stroke ~ ., data = train_smote,
                          method = "svmLinear", trControl = ctrl, metric = "ROC")

set.seed(42)
model_svm_rbf <- train(stroke ~ ., data = train_smote,
                       method = "svmRadial", trControl = ctrl, metric = "ROC")

# -----------------------------
# TRUE NYSTRÖM APPROXIMATION
# -----------------------------

# Convert to matrix
x_train <- as.matrix(train_smote[, -which(names(train_smote) == "stroke")])
y_train <- train_smote$stroke
x_test  <- as.matrix(test_processed[, -which(names(test_processed) == "stroke")])

set.seed(42)
m <- 300  # number of landmark points
idx <- sample(1:nrow(x_train), m)
landmarks <- x_train[idx, ]

# RBF kernel
rbf_kernel <- function(x, y, sigma = 0.05) {
  as.matrix(exp(-sigma * (as.matrix(dist(rbind(x,y)))[1:nrow(x), (nrow(x)+1):(nrow(x)+nrow(y))]^2)))
}

# Kernel matrices
K_mm <- rbf_kernel(landmarks, landmarks)
K_nm <- rbf_kernel(x_train, landmarks)

# Eigen decomposition
eig <- eigen(K_mm)
U <- eig$vectors
S <- diag(1 / sqrt(eig$values + 1e-8))

# Feature mapping
Z_train <- K_nm %*% U %*% S

# Transform test
K_test <- rbf_kernel(x_test, landmarks)
Z_test <- K_test %*% U %*% S

# Train linear SVM
model_nystrom <- svm(
  x = Z_train,
  y = y_train,
  kernel = "linear",
  probability = TRUE
)

# -----------------------------
# Prediction + METRICS
# -----------------------------
predict_model <- function(model, name, is_kernlab = FALSE) {

  if (is_kernlab) {
    pred_obj <- predict(model, Z_test, probability = TRUE)
    probs <- attr(pred_obj, "probabilities")[, "Stroke"]
    preds <- ifelse(probs > 0.5, "Stroke", "No_Stroke")
    preds <- factor(preds, levels = c("Stroke", "No_Stroke"))
  } else {
    preds <- predict(model, test_processed)
    probs <- predict(model, test_processed, type = "prob")[, "Stroke"]
  }

  cm <- confusionMatrix(preds, test_data$stroke)
  roc_obj <- roc(response = test_data$stroke, predictor = probs)

  precision <- cm$byClass["Pos Pred Value"]
  recall    <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  f1 <- (2 * precision * recall) / (precision + recall)

  data.frame(
    Model = name,
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Recall = recall,
    Specificity = specificity,
    Precision = precision,
    F1 = f1,
    AUC = as.numeric(auc(roc_obj))
  )
}

# -----------------------------
# Collect results
# -----------------------------
results_df <- bind_rows(
  predict_model(model_log, "Logistic Regression"),
  predict_model(model_rf, "Random Forest"),
  predict_model(model_svm_linear, "Linear SVM"),
  predict_model(model_svm_rbf, "RBF SVM"),
  predict_model(model_nystrom, "Nyström SVM", TRUE)
)

# -----------------------------
# PRINT + SAVE
# -----------------------------
print(results_df)

write.csv(results_df, "outputs/model_results_smote.csv", row.names = FALSE)

cat("\nSaved to: outputs/model_results_smote.csv\n")