library(tidyverse)
library(caret)
library(e1071)
library(randomForest)
library(pROC)

# 1. Load preprocessed data
train_data <- read_csv("outputs/train_data.csv", show_col_types = FALSE)
test_data  <- read_csv("outputs/test_data.csv", show_col_types = FALSE)

# 2. Fix target labels
train_data$stroke <- ifelse(train_data$stroke == "Stroke", "Stroke", "No_Stroke")
test_data$stroke  <- ifelse(test_data$stroke == "Stroke", "Stroke", "No_Stroke")

train_data$stroke <- factor(train_data$stroke, levels = c("Stroke", "No_Stroke"))
test_data$stroke  <- factor(test_data$stroke, levels = c("Stroke", "No_Stroke"))

# 3. Convert categorical predictors
factor_cols <- c(
  "gender", "ever_married", "work_type",
  "Residence_type", "smoking_status",
  "hypertension", "heart_disease"
)

train_data[factor_cols] <- lapply(train_data[factor_cols], factor)
test_data[factor_cols]  <- lapply(test_data[factor_cols], factor)

# 4. Train control
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# 5. Logistic Regression
set.seed(42)
model_log <- train(
  stroke ~ ., data = train_data,
  method = "glm", family = "binomial",
  trControl = ctrl, metric = "ROC"
)

# 6. Random Forest
set.seed(42)
model_rf <- train(
  stroke ~ ., data = train_data,
  method = "rf",
  trControl = ctrl, metric = "ROC",
  ntree = 200
)

# 7. Linear SVM
set.seed(42)
model_svm_linear <- train(
  stroke ~ ., data = train_data,
  method = "svmLinear",
  trControl = ctrl, metric = "ROC"
)

# 8. RBF SVM
set.seed(42)
model_svm_rbf <- train(
  stroke ~ ., data = train_data,
  method = "svmRadial",
  trControl = ctrl, metric = "ROC"
)

# 9. NYSTRÖM SVM (FIXED)

# Combine to ensure same columns
combined <- bind_rows(train_data, test_data)

# Create consistent numeric matrix
full_matrix <- model.matrix(stroke ~ . - 1, data = combined)

# Split back
x_train <- full_matrix[1:nrow(train_data), ]
x_test  <- full_matrix[(nrow(train_data)+1):nrow(full_matrix), ]

y_train <- train_data$stroke

set.seed(42)
m <- 300
idx <- sample(1:nrow(x_train), m)
landmarks <- x_train[idx, ]

rbf_kernel <- function(x, y, sigma = 0.05) {
  dist_matrix <- as.matrix(dist(rbind(x, y)))
  K <- exp(-sigma * dist_matrix^2)
  K[1:nrow(x), (nrow(x)+1):(nrow(x)+nrow(y))]
}

K_mm <- rbf_kernel(landmarks, landmarks)
K_nm <- rbf_kernel(x_train, landmarks)

eig <- eigen(K_mm)
U <- eig$vectors
S <- diag(1 / sqrt(eig$values + 1e-8))

Z_train <- K_nm %*% U %*% S
K_test <- rbf_kernel(x_test, landmarks)
Z_test <- K_test %*% U %*% S

model_nystrom <- svm(
  x = Z_train,
  y = y_train,
  kernel = "linear",
  probability = TRUE
)

# 10. Prediction helper
predict_model <- function(model, name, is_nystrom = FALSE) {
  
  if (is_nystrom) {
    pred_obj <- predict(model, Z_test, probability = TRUE)
    probs <- attr(pred_obj, "probabilities")[, "Stroke"]
    preds <- ifelse(probs > 0.5, "Stroke", "No_Stroke")
    preds <- factor(preds, levels = c("Stroke", "No_Stroke"))
  } else {
    preds <- predict(model, test_data)
    probs <- predict(model, test_data, type = "prob")[, "Stroke"]
  }
  
  cm <- confusionMatrix(preds, test_data$stroke)
  roc_obj <- roc(response = test_data$stroke, predictor = probs)
  
  list(
    name = name,
    confusion = cm,
    AUC = auc(roc_obj)
  )
}

# 11. Collect results
results <- list(
  predict_model(model_log, "Logistic Regression"),
  predict_model(model_rf, "Random Forest"),
  predict_model(model_svm_linear, "Linear SVM"),
  predict_model(model_svm_rbf, "RBF SVM"),
  predict_model(model_nystrom, "Nyström SVM", TRUE)
)

# 12. Print results
for (res in results) {
  cat("\n=========================\n")
  cat("Model:", res$name, "\n")
  print(res$confusion)
  cat("AUC:", as.numeric(res$AUC), "\n")
}

# 13. Save results to CSV
results_df <- bind_rows(lapply(results, function(res) {
  cm <- res$confusion
  
  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  specificity <- cm$byClass["Specificity"]
  f1 <- (2 * precision * recall) / (precision + recall)
  
  data.frame(
    Model = res$name,
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Recall = recall,
    Specificity = specificity,
    Precision = precision,
    F1 = f1,
    AUC = as.numeric(res$AUC)
  )
}))

write.csv(results_df, "outputs/model_results.csv", row.names = FALSE)

cat("\nSaved to: outputs/model_results.csv\n")

# 14. Plot model comparison chart
library(ggplot2)
library(tidyr)

# Reshape to long format for plotting
results_long <- results_df %>%
  select(Model, AUC, Accuracy, Recall, Specificity, Precision, F1) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

# Plot
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  ylim(0, 1) +
  labs(
    title = "Model Comparison — Stroke Prediction",
    x = NULL,
    y = "Score",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1),
    plot.title = element_text(size = 14, face = "bold")
  )

# Save chart to folder
library(ggplot2)
library(tidyr)

# Create folder if it doesn't exist
if (!dir.exists("charts")) {
  dir.create("charts")
}

# Prepare data
results_long <- results_df %>%
  pivot_longer(
    cols = c(Accuracy, AUC, F1),
    names_to = "Metric",
    values_to = "Value"
  )

# Create plot
p <- ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(
    title = "Model Performance Comparison",
    x = "Model",
    y = "Score"
  ) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Save plot
ggsave(
  filename = "charts/model_baseline_comparison.png",
  plot = p,
  width = 8,
  height = 5
)

cat("Chart saved to: charts/model_baseline_comparison.png\n")