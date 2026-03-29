library(tidyverse)

stroke_data <- read_csv("data/healthcare-dataset-stroke-data.csv", show_col_types = FALSE) %>%
  mutate(
    bmi = na_if(bmi, "N/A"),
    bmi = as.numeric(bmi),
    stroke_label = if_else(stroke == 1, "Stroke", "No Stroke")
  )

# -----------------------------
# 1. Class distribution plot
# -----------------------------
p1 <- ggplot(stroke_data, aes(x = stroke_label, fill = stroke_label)) +
  geom_bar() +
  labs(
    title = "Class Distribution",
    x = "",
    y = "Count"
  ) +
  scale_fill_manual(values = c("No Stroke" = "#2C3E50", "Stroke" = "#C0392B")) +
  theme_minimal() +
  theme(legend.position = "none")

print(p1)

ggsave("figures/class_distribution.png", plot = p1, width = 6, height = 4, dpi = 300)

# -----------------------------
# 2. Age distribution by stroke
# -----------------------------
p2 <- ggplot(stroke_data, aes(x = age, fill = stroke_label, color = stroke_label)) +
  geom_density(alpha = 0.35) +
  labs(
    title = "Age Distribution by Stroke Outcome",
    x = "Age",
    y = "Density",
    fill = "Outcome",
    color = "Outcome"
  ) +
  scale_fill_manual(values = c("No Stroke" = "#2C3E50", "Stroke" = "#C0392B")) +
  scale_color_manual(values = c("No Stroke" = "#2C3E50", "Stroke" = "#C0392B")) +
  theme_minimal()

print(p2)
ggsave("figures/age_distribution.png", p2, width = 6, height = 4, dpi = 300)

# -----------------------------
# 3. Glucose level by stroke
# -----------------------------
p3 <- ggplot(stroke_data, aes(x = factor(stroke), y = avg_glucose_level, fill = factor(stroke))) +
  geom_boxplot() +
  labs(
    title = "Average Glucose Level by Stroke Outcome",
    x = "Stroke",
    y = "Average Glucose Level",
    fill = "Stroke"
  ) +
  scale_x_discrete(labels = c("0" = "No Stroke", "1" = "Stroke")) +
  scale_fill_manual(values = c("0" = "#2C3E50", "1" = "#C0392B")) +
  theme_minimal()

print(p3)
ggsave("figures/glucose_boxplot.png", p3, width = 6, height = 4, dpi = 300)