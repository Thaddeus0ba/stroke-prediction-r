library(tidyverse)

# Load results
baseline <- read_csv("outputs/model_results.csv")
smote    <- read_csv("outputs/model_results_smote.csv")
up       <- read_csv("outputs/model_results_upsampling.csv")

# Add labels
baseline$Method <- "Baseline"
smote$Method    <- "SMOTE"
up$Method       <- "Upsampling"

# Combine
all_results <- bind_rows(baseline, smote, up)

# Create folder
if (!dir.exists("charts")) dir.create("charts")

# Plot
library(ggplot2)

p <- ggplot(all_results, aes(x = Model, y = AUC, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(
    title = "Final Model Comparison",
    x = "Model",
    y = "AUC"
  ) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

# Save
ggsave("charts/model_performance_summary.png", p, width = 10, height = 6, dpi = 300)

cat("DONE → charts/model_performance_summary.png\n")