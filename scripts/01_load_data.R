library(tidyverse)

stroke_data <- read_csv("data/healthcare-dataset-stroke-data.csv", show_col_types = FALSE)

# Convert BMI from character to numeric
stroke_data <- stroke_data %>%
  mutate(
    bmi = na_if(bmi, "N/A"),
    bmi = as.numeric(bmi)
  )

# Check structure again
glimpse(stroke_data)

# Missing values
colSums(is.na(stroke_data))

