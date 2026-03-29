library(tidyverse)
library(caret)

# Load and clean data
stroke_data <- read_csv("data/healthcare-dataset-stroke-data.csv", show_col_types = FALSE) %>%
  mutate(
    bmi = na_if(bmi, "N/A"),
    bmi = as.numeric(bmi),
    stroke = factor(stroke, levels = c(0, 1), labels = c("No Stroke", "Stroke"))
  )

# Impute missing BMI with median
median_bmi <- median(stroke_data$bmi, na.rm = TRUE)
stroke_data <- stroke_data %>%
  mutate(bmi = if_else(is.na(bmi), median_bmi, bmi))

# Convert categorical variables to factors
stroke_data <- stroke_data %>%
  mutate(
    gender = as.factor(gender),
    ever_married = as.factor(ever_married),
    work_type = as.factor(work_type),
    Residence_type = as.factor(Residence_type),
    smoking_status = as.factor(smoking_status),
    hypertension = as.factor(hypertension),
    heart_disease = as.factor(heart_disease)
  )

# Remove ID column
stroke_data <- stroke_data %>%
  select(-id)

# Train/test split (stratified)
set.seed(42)
train_index <- createDataPartition(stroke_data$stroke, p = 0.8, list = FALSE)

train_data <- stroke_data[train_index, ]
test_data  <- stroke_data[-train_index, ]

# Save split datasets
write_csv(train_data, "outputs/train_data.csv")
write_csv(test_data, "outputs/test_data.csv")

# Quick checks
cat("Train size:", nrow(train_data), "\n")
cat("Test size:", nrow(test_data), "\n")
cat("Train stroke distribution:\n")
print(table(train_data$stroke))
cat("Test stroke distribution:\n")
print(table(test_data$stroke))

# 