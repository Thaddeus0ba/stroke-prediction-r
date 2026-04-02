# ================================
# FINAL SUBMISSION CODE (UPSAMPLING + SVMs)
# ================================

library(caret)
library(e1071)
library(PRROC)
library(kernlab)

cat("Script started...\n")

# -------------------------------
# LOAD DATA
# -------------------------------
df <- read.csv("healthcare-dataset-stroke-data.csv")

# -------------------------------
# CLEANING
# -------------------------------
df <- df[, !(names(df) %in% c("id"))]

df$bmi <- as.numeric(df$bmi)
df$age <- as.numeric(df$age)
df$avg_glucose_level <- as.numeric(df$avg_glucose_level)

df$bmi[is.na(df$bmi)] <- median(df$bmi, na.rm = TRUE)

df$gender[df$gender == "Other"] <- NA

df$gender <- as.factor(df$gender)
df$ever_married <- as.factor(df$ever_married)
df$work_type <- as.factor(df$work_type)
df$Residence_type <- as.factor(df$Residence_type)
df$smoking_status <- as.factor(df$smoking_status)

df$stroke <- factor(df$stroke, levels = c(0,1))

df <- na.omit(df)

# -------------------------------
# SPLIT
# -------------------------------
set.seed(42)
idx <- createDataPartition(df$stroke, p = 0.8, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

# -------------------------------
# SCALE NUMERIC
# -------------------------------
num_cols <- c("age","avg_glucose_level","bmi")

pre <- preProcess(train[, num_cols], method = c("center","scale"))
train[, num_cols] <- predict(pre, train[, num_cols])
test[, num_cols]  <- predict(pre, test[, num_cols])

# -------------------------------
# UPSAMPLING (BEST METHOD)
# -------------------------------
train_bal <- upSample(
  x = train[, -which(names(train) == "stroke")],
  y = train$stroke,
  yname = "stroke"
)

cat("Training models...\n")

# -------------------------------
# MODELS
# -------------------------------

# Linear SVM (BEST)
svm_lin <- svm(
  stroke ~ ., data = train_bal,
  kernel = "linear", cost = 1,
  probability = TRUE
)

lin_prob <- attr(predict(svm_lin, test, probability=TRUE), "probabilities")[,2]

# Nyström SVM
svm_nys <- ksvm(
  stroke ~ ., data = train_bal,
  kernel = "rbfdot", kpar=list(sigma=0.08),
  prob.model = TRUE
)

nys_prob <- predict(svm_nys, test, type="probabilities")[,2]

# -------------------------------
# EVALUATION
# -------------------------------
evaluate <- function(true, prob, name){

  true <- as.numeric(as.character(true))
  pred <- ifelse(prob > 0.2, 1, 0)

  cm <- confusionMatrix(
    factor(pred),
    factor(true),
    positive="1"
  )

  precision <- cm$byClass["Pos Pred Value"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)

  pr <- pr.curve(
    scores.class0 = prob[true==1],
    scores.class1 = prob[true==0]
  )

  cat("\n====================\n")
  cat("Model:", name, "\n")
  cat("====================\n")
  print(cm)
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1 Score:", f1, "\n")
  cat("PR-AUC:", pr$auc.integral, "\n")
}

cat("\nFINAL RESULTS\n")

evaluate(test$stroke, lin_prob, "Linear SVM (BEST)")
evaluate(test$stroke, nys_prob, "Nyström SVM")

cat("\nDONE ✅\n")