# Install and Load Necessary Libraries
install.packages(c("dplyr", "ggplot2", "leaps", "glmnet", "caret", "class", "pROC", "readr", "MASS"))
library(dplyr)
library(ggplot2)
library(leaps)
library(glmnet)
library(caret)
library(class)
library(pROC)
library(readr)
library(MASS)


### PART 2 ###
# Load the dataset and provide summary (Change file location as necessary)
data <- read.csv("~/Documents/Lectures/MIS649/medical_insurance.csv", sep = ",")
summary(data)
str(data)
colSums(is.na(data))


# Generate dummy variables for the 'region' categorical variable
data <- data.frame(data, model.matrix(~ region - 1, data = data))


# Create Risk Level Variable
data$risk_level <- cut(data$charges, breaks = c(-Inf, 5000, 15000, Inf), 
                       labels = c("Low Risk", "Medium Risk", "High Risk"))
table(data$risk_level)


# Reusable plot function for histograms
plot_histogram <- function(df, column, binwidth, fill_color, title) {
  ggplot(df, aes_string(x = column)) + 
    geom_histogram(binwidth = binwidth, color = "black", fill = fill_color) +
    theme_minimal() +
    labs(title = title, x = column, y = "Frequency")
}


# Histograms for numerical variables
plot_histogram(data, "age", 5, "lightblue", "Distribution of Age")
plot_histogram(data, "bmi", 1, "lightgreen", "Distribution of BMI")
plot_histogram(data, "charges", 2000, "lightpink", "Distribution of Charges")


# Bar plot function
plot_bar <- function(df, column, fill_color, title) {
  ggplot(df, aes_string(x = column)) +
    geom_bar(color = "black", fill = fill_color) +
    theme_minimal() +
    labs(title = title, x = column, y = "Count")
}


# Bar plots for categorical variables
plot_bar(data, "sex", "orange", "Distribution of Sex")
plot_bar(data, "smoker", "purple", "Distribution of Smoker Status")
plot_bar(data, "region", "blue", "Distribution of Region")
plot_bar(data, "risk_level", "skyblue", "Distribution of Risk Levels")


## Bivariate Analysis
data$risk_level_numeric <- as.numeric(data$risk_level)
numerical_data <- data %>% select(age, bmi, children, risk_level_numeric)
cor_matrix <- cor(numerical_data)
cor_matrix


# Boxplot function
plot_boxplot <- function(df, x, y, fill, title, y_label) {
  ggplot(df, aes_string(x = x, y = y, fill = fill)) +
    geom_boxplot() +
    theme_minimal() +
    labs(title = title, x = x, y = y_label)
}
plot_boxplot(data, "risk_level", "age", "risk_level", "Age vs. Risk Level", "Age")
plot_boxplot(data, "risk_level", "bmi", "risk_level", "BMI vs. Risk Level", "BMI")
plot_boxplot(data, "risk_level", "children", "risk_level", "Number of Children vs. Risk Level", "Children")


# Stacked bar plots for relationships
stacked_bar_plot <- function(df, x, fill, title) {
  ggplot(df, aes_string(x = x, fill = fill)) +
    geom_bar(position = "fill", color = "black") +
    labs(title = title, x = x, y = "Proportion", fill = fill) +
    theme_minimal()
}
stacked_bar_plot(data, "risk_level", "smoker", "Smoker Status vs. Risk Level")
stacked_bar_plot(data, "risk_level", "region", "Region vs. Risk Level")


# Scatterplots
scatter_plot <- function(df, x, y, color, title, x_label, y_label) {
  ggplot(df, aes_string(x = x, y = y, color = color)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "lm", se = FALSE, color = "black") +
    labs(title = title, x = x_label, y = y_label, color = color) +
    theme_minimal()
}


scatter_plot(data, "bmi", "age", "risk_level", "BMI vs. Age by Risk Level", "BMI", "Age")
scatter_plot(data, "age", "charges", "risk_level", "Age vs. Charges by Risk Level", "Age", "Charges")
scatter_plot(data, "bmi", "charges", "risk_level", "BMI vs. Charges by Risk Level", "BMI", "Charges")


# Density plots
density_plot <- function(df, x, fill, title) {
  ggplot(df, aes_string(x = x, fill = fill)) +
    geom_density(alpha = 0.5) +
    theme_minimal() +
    labs(title = title, x = x, y = "Density", fill = fill)
}


density_plot(data, "age", "smoker", "Age Distribution by Smoking Status")
density_plot(data, "bmi", "smoker", "BMI Distribution by Smoking Status")


# Data Cleaning
data$bmi <- pmin(data$bmi, 50)
data$log_charges <- log(data$charges)
data$bmi_smoker <- data$bmi * as.numeric(data$smoker == "yes")
data$age_smoker <- data$age * as.numeric(data$smoker == "yes")


normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
data$age_norm <- normalize(data$age)
data$bmi_norm <- normalize(data$bmi)


# Summary of the cleaned data
summary(data)


### PART 3 ###


# Set seed and split data into training and test sets
set.seed(123)
train_index <- createDataPartition(data$log_charges, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]
cat("Training data:", nrow(train_data), "rows; Test data:", nrow(test_data), "rows\n")


### Regression Section ###
# Subset selection using training data
best_subset <- regsubsets(log_charges ~ age_norm + bmi_norm + children + sex + region + bmi_smoker + age_smoker,
                          data = train_data, nvmax = 8)
subset_summary <- summary(best_subset)


# Extract best subset predictors and Adjusted R^2
best_model_index <- which.max(subset_summary$adjr2)
best_model <- (subset_summary$which[best_model_index, -1])
best_model_which <- names(which(best_model))
adjusted_r2 <- subset_summary$adjr2[best_model_index]
cat("Best Subset Model:\n", best_model_which, "\nAdjusted R^2:", adjusted_r2, "\n")




# Fit and summarize best subset model
best_subset_model <- lm(as.formula(paste("log_charges ~", paste(best_model, collapse = " + "))), data = train_data)
summary(best_subset_model)


# Predictions and Test MSE for Subset Model
subset_predictions <- predict(best_subset_model, newdata = test_data)
subset_test_mse <- mean((test_data$log_charges - subset_predictions)^2)
cat("Test MSE for Subset Model:", subset_test_mse, "\n")


# Residuals and fitted values
subset_residuals <- residuals(best_subset_model)
subset_fitted <- fitted(best_subset_model)


# Plot Residuals vs Predicted Values
plot(subset_fitted, subset_residuals,
     main = "Residuals vs Predicted Values (Subset Model)",
     xlab = "Predicted Values", ylab = "Residuals", pch = 20, col = "blue")
abline(h = 0, col = "red", lty = 2)


hist(subset_residuals, breaks = 20, col = "lightblue", 
     main = "Histogram of Residuals (Subset Model)", xlab = "Residuals")


qqnorm(subset_residuals, main = "QQ Plot of Residuals (Subset Model)")
qqline(subset_residuals, col = "red", lty = 2)


# Prepare model matrix with dummy variables
X_train <- model.matrix(log_charges ~ age_norm + bmi_norm + children + sex + 
                          regionsoutheast + regionsouthwest + bmi_smoker + age_smoker, train_data)[, -1]
X_test <- model.matrix(log_charges ~ age_norm + bmi_norm + children + sex + 
                         regionsoutheast + regionsouthwest + bmi_smoker + age_smoker, test_data)[, -1]
y_train <- train_data$log_charges
y_test <- test_data$log_charges


# Fit Ridge regression with cross-validation
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0, lambda = 10^seq(10, -2, length = 100), standardize = TRUE)


# Optimal lambda for Ridge
ridge_lambda <- ridge_model$lambda.min


# Predict on test set
ridge_predictions <- predict(ridge_model, s = ridge_lambda, newx = X_test)


# Calculate Test MSE
ridge_test_mse <- mean((y_test - ridge_predictions)^2)
cat("Test MSE for Ridge Regression:", ridge_test_mse)


# Extract Ridge coefficients at optimal lambda
ridge_coefficients <- as.data.frame(as.matrix(coef(ridge_model, s = ridge_lambda)))
colnames(ridge_coefficients) <- c("Ridge Coefficient")
ridge_coefficients$Predictor <- rownames(ridge_coefficients)
rownames(ridge_coefficients) <- NULL
cat("Ridge Regression Coefficients:\n", ridge_coefficients, "\n")


# Fit Lasso regression with cross-validation
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1, lambda = 10^seq(10, -2, length = 100), standardize = TRUE)


# Optimal lambda for Lasso
lasso_lambda <- lasso_model$lambda.min


# Predict on test set
lasso_predictions <- predict(lasso_model, s = lasso_lambda, newx = X_test)


# Calculate Test MSE
lasso_test_mse <- mean((y_test - lasso_predictions)^2)
cat("Test MSE for Lasso Regression:", lasso_test_mse)


# Extract Lasso coefficients at optimal lambda
lasso_coefficients <- as.data.frame(as.matrix(coef(lasso_model, s = lasso_lambda)))
colnames(lasso_coefficients) <- c("Lasso Coefficient")
lasso_coefficients$Predictor <- rownames(lasso_coefficients)
rownames(lasso_coefficients) <- NULL


cat("Lasso Regression Coefficients:", lasso_coefficients)


# Residual diagnostics for Ridge
ridge_residuals <- y_test - ridge_predictions


# Residual Plot
plot(ridge_predictions, ridge_residuals,
     main = "Residuals vs Predicted Values (Ridge Model)",
     xlab = "Predicted Values", ylab = "Residuals", pch = 20, col = "blue")
abline(h = 0, col = "red", lty = 2)


# Histogram of Residuals
hist(ridge_residuals, breaks = 20, col = "lightblue", 
     main = "Histogram of Residuals (Ridge Model)", xlab = "Residuals")


# Residual diagnostics for Lasso
lasso_residuals <- y_test - lasso_predictions


# Residual Plot
plot(lasso_predictions, lasso_residuals,
     main = "Residuals vs Predicted Values (Lasso Model)",
     xlab = "Predicted Values", ylab = "Residuals", pch = 20, col = "blue")
abline(h = 0, col = "red", lty = 2)


# Histogram of Residuals
hist(lasso_residuals, breaks = 20, col = "lightblue", 
     main = "Histogram of Residuals (Lasso Model)", xlab = "Residuals")


### Classification Section ###


library(caret)
library(dplyr)
# Load required libraries
library(dplyr)
library(caret)
library(nnet)
# Load and prepare the dataset
data <- read.csv("Book3.csv")
# Create risk_level column based on 'charges'
data$risk_level <- cut(
  data$charges,
  breaks = c(-Inf, 5000, 15000, Inf),
  labels = c("Low Risk", "Medium Risk", "High Risk")
)
data$risk_level <- as.factor(data$risk_level)


# Remove unnecessary columns and ensure categorical variables are factors
data$sex <- as.factor(data$sex)
data$smoker <- as.factor(data$smoker)
data$region <- as.factor(data$region)
# Remove rows with missing values
data <- na.omit(data)
# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(data$risk_level, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]
# Train logistic regression model
train_control <- trainControl(method = "cv", number = 10)
logistic_model <- train(
  risk_level ~ ., 
  data = train, 
  method = "multinom",  # Multinomial logistic regression
  trControl = train_control,
  preProcess = c("center", "scale")
)


# Print model coefficients
coefficients <- summary(logistic_model$finalModel)$coefficients
print("Coefficients:")
print(coefficients)
# Predict on the test set
logistic_predictions <- predict(logistic_model, newdata = test)
# Evaluate the model
conf_matrix <- confusionMatrix(logistic_predictions, test$risk_level)
print("Confusion Matrix and Statistics:")
print(conf_matrix)
# Calculate test error
test_error <- 1 - conf_matrix$overall["Accuracy"]
cat("Cross-Validated Test Error:", test_error, "\n")
# Sensitivity and Specificity
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
cat("Sensitivity (Recall):\n")
print(sensitivity)
cat("Specificity:\n")
print(specificity)


# 2nd Part


library(MASS)
library(caret)
library(dplyr)
data <- read.csv("Book3.csv")


# Create risk_level column based on 'charges'
data$risk_level <- cut(
  data$charges,
  breaks = c(-Inf, 5000, 15000, Inf),
  labels = c("Low Risk", "Medium Risk", "High Risk")
)
data$risk_level <- as.factor(data$risk_level)
# Ensure categorical variables are factors
data$sex <- as.factor(data$sex)
data$smoker <- as.factor(data$smoker)
data$region <- as.factor(data$region)
# Remove rows with missing values
data <- na.omit(data)
# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(data$risk_level, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]
# Train the LDA model
lda_model <- lda(risk_level ~ age + bmi + children + sex + smoker + region, data = train)
# Print LDA coefficients
print("LDA Coefficients:")
print(lda_model$scaling)
# Make predictions on the test set
lda_predictions <- predict(lda_model, newdata = test)
predicted_classes <- lda_predictions$class
# Confusion Matrix and Statistics
conf_matrix <- confusionMatrix(predicted_classes, test$risk_level)
print("Confusion Matrix and Statistics:")
print(conf_matrix)


# Cross-validated test error
train_control <- trainControl(method = "cv", number = 10)
lda_cv <- train(
  risk_level ~ age + bmi + children + sex + smoker + region,
  data = train,
  method = "lda",
  trControl = train_control
)
cv_test_error <- 1 - max(lda_cv$results$Accuracy)
cat("Cross-Validated Test Error:", cv_test_error, "\n")
# Sensitivity and Specificity
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
cat("Sensitivity (Recall):\n")
print(sensitivity)
cat("Specificity:\n")
print(specificity)


#Part 3


# Create risk_level column based on 'charges'
data$risk_level <- cut(
  data$charges,
  breaks = c(-Inf, 5000, 15000, Inf),
  labels = c("Low Risk", "Medium Risk", "High Risk")
)
data$risk_level <- as.factor(data$risk_level)


# Ensure categorical variables are factors
data$sex <- as.factor(data$sex)
data$smoker <- as.factor(data$smoker)
data$region <- as.factor(data$region)
# Remove rows with missing values
data <- na.omit(data)
# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(data$risk_level, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]


# Preprocess: Normalize numerical features and dummy-encode categorical features
preprocess <- preProcess(train[, -which(colnames(train) == "risk_level")], method = c("center", "scale"))
X_train <- predict(preprocess, train[, -which(colnames(train) == "risk_level")])
X_test <- predict(preprocess, test[, -which(colnames(test) == "risk_level")])
# Ensure predictors are data frames
X_train <- as.data.frame(X_train)
X_test <- as.data.frame(X_test)
# Define the target variable
y_train <- train$risk_level
y_test <- test$risk_level
# Find the optimal K using cross-validation
train_control <- trainControl(method = "cv", number = 10)
knn_model <- train(
  risk_level ~ ., 
  data = train, 
  method = "knn", 
  trControl = train_control,
  preProcess = c("center", "scale"),
  tuneLength = 20 # Test k values from 1 to 20
)
# Print model details
print(knn_model)
# Optimal value of K
optimal_k <- knn_model$bestTune$k
cat("Optimal K:", optimal_k, "\n")
# Test the model on the test set
knn_predictions <- predict(knn_model, newdata = test)
# Evaluate the model
conf_matrix <- confusionMatrix(knn_predictions, test$risk_level)
print("Confusion Matrix and Statistics:")
print(conf_matrix)
# Cross-validated test error
cv_test_error <- 1 - max(knn_model$results$Accuracy)
cat("Cross-Validated Test Error:", cv_test_error, "\n")
# Sensitivity and Specificity
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
cat("Sensitivity (Recall):\n")
print(sensitivity)


cat("Specificity:\n")
print(specificity)
