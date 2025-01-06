
##### NOTE: Please run each block seperately as each member did their code individually and in their own way.
## For SVM and logistic regression: data.csv was made for better exploration of data.
## I'm attaching csv file, please use that for block-2 and bloack-3.

## FINAL REPORT R CODE 
## SAAD CHADRAWALA AND MIHIR THAKKAR
## STATS 385

##################BLOCK-1 ################################

# Load all the necessary libraries
library(rpart)        # For building recursive partitioning decision trees.
library(rpart.plot)   # For visualizing decision trees created by rpart.
library(caret)        # For data partitioning, modeling, and evaluating the model.

# Set working directory
setwd("/Users/saadchadrawala/Documents/Stats 385")

# Load and explore the dataset
wdbc.data <- read.csv("wdbc.data", header = FALSE)

# View the structure and first few rows of the dataset
str(wdbc.data)    # To view the structure of the data
head(wdbc.data)   # To view the first few rows

# Summary statistics of the dataset
summary(wdbc.data)   # Provides a summary of each column

# Assign column names
colnames(wdbc.data) <- c("ID", "Diagnosis", paste0(rep(c("Mean", "SE", "Worst"), each = 10),
                                                   c("Radius", "Texture", "Perimeter", "Area", "Smoothness", "Compactness",
                                                     "Concavity", "ConcavePoints", "Symmetry", "FractalDimension")))

# Convert Diagnosis to a factor (for classification)
wdbc.data$Diagnosis <- as.factor(wdbc.data$Diagnosis)

# Split the data into training and test sets
set.seed(123)   # Set random seed for reproducibility

trainIndex <- createDataPartition(wdbc.data$Diagnosis, p = 0.7, list = FALSE)
trainData <- wdbc.data[trainIndex, ]  # Training data
testData <- wdbc.data[-trainIndex, ]  # Test data

######################### Decision Tree Model Code Starts Here ------------------------------------------------------------
# Step 1: Train an Untuned Decision Tree Model
model <- rpart(Diagnosis ~ ., data = trainData, method = "class") 
# Builds a decision tree model to predict 'Diagnosis' using all other variables as predictors. 
# The 'method = "class"' specifies a classification task.

# Step 2: Plot the Untuned Decision Tree
rpart.plot(model) 
# Visualizes the decision tree using the rpart.plot package.

# Step 3: Make Predictions on the Test Set (Untuned Model)
predictions_untuned <- predict(model, testData, type = "class") 
# Predicts the class labels for the test set using the trained untuned model.

# Step 4: Evaluate the Untuned Model Using a Confusion Matrix
confusion_matrix_untuned <- confusionMatrix(predictions_untuned, testData$Diagnosis) 
# Computes a confusion matrix comparing predicted and actual diagnoses for the test set (untuned model).

# Print confusion matrix for the untuned model
print("Confusion Matrix (Untuned Model):")
print(confusion_matrix_untuned)

# Step 5: Tune the Model by Adjusting the Complexity Parameter (cp)
model_tuned <- rpart(Diagnosis ~ ., data = trainData, method = "class", cp = 0.01) 
# Retrains the decision tree model with a lower cp value (0.01) to allow a more complex tree.

# Step 6: Plot the Tuned Decision Tree
rpart.plot(model_tuned) 
# Visualizes the new, tuned decision tree.

# Step 7: Make Predictions on the Test Set (Tuned Model)
predictions_tuned <- predict(model_tuned, testData, type = "class") 
# Predicts the class labels for the test set using the trained tuned model.

# Step 8: Evaluate the Tuned Model Using a Confusion Matrix
confusion_matrix_tuned <- confusionMatrix(predictions_tuned, testData$Diagnosis) 
# Computes a confusion matrix comparing predicted and actual diagnoses for the test set (tuned model).

# Print confusion matrix for the tuned model
print("Confusion Matrix (Tuned Model):")
print(confusion_matrix_tuned)

################### RANDOM FOREST ###################
# Install required packages if you haven't already
install.packages("randomForest")
install.packages("caret")

# Load the libraries
library(randomForest)
library(caret)
# Assuming you've already loaded and preprocessed the data as shown in your previous steps

# Train a Random Forest model
set.seed(123)  # Set seed for reproducibility
rf_model <- randomForest(Diagnosis ~ ., data = trainData, importance = TRUE)

# Print the model summary
print(rf_model)

# Plot the importance of features
plot(rf_model)

# Make predictions using the trained Random Forest model
rf_predictions <- predict(rf_model, testData)

# Evaluate model performance using a confusion matrix
rf_conf_matrix <- confusionMatrix(rf_predictions, testData$Diagnosis)
print(rf_conf_matrix)
# Confusion Matrix output
# You can also extract individual metrics like this:
accuracy_rf <- rf_conf_matrix$overall["Accuracy"]
sensitivity_rf <- rf_conf_matrix$byClass["Sensitivity"]
specificity_rf <- rf_conf_matrix$byClass["Specificity"]

# Print the evaluation metrics
cat("Accuracy: ", accuracy_rf, "\n")
cat("Sensitivity: ", sensitivity_rf, "\n")
cat("Specificity: ", specificity_rf, "\n")
# Tune the Random Forest model (e.g., number of trees and features considered at each split)
set.seed(123)
tuned_rf_model <- randomForest(Diagnosis ~ ., data = trainData, ntree = 1000, mtry = 5, importance = TRUE)

# Evaluate the tuned model
tuned_rf_predictions <- predict(tuned_rf_model, testData)
tuned_rf_conf_matrix <- confusionMatrix(tuned_rf_predictions, testData$Diagnosis)

# Print the confusion matrix for the tuned model
print(tuned_rf_conf_matrix)

# Extract metrics for the tuned model
accuracy_tuned_rf <- tuned_rf_conf_matrix$overall["Accuracy"]
sensitivity_tuned_rf <- tuned_rf_conf_matrix$byClass["Sensitivity"]
specificity_tuned_rf <- tuned_rf_conf_matrix$byClass["Specificity"]

# Print the evaluation metrics of the tuned model
cat("Tuned Random Forest Accuracy: ", accuracy_tuned_rf, "\n")
cat("Tuned Random Forest Sensitivity: ", sensitivity_tuned_rf, "\n")
cat("Tuned Random Forest Specificity: ", specificity_tuned_rf, "\n")

#################33 KNN method -------------------------------
# Load necessary libraries
library(class)    # For knn
library(caret)    # For confusionMatrix


# Split the data into train and test sets using the createDataPartition function
set.seed(123)  # To ensure reproducibility of the split
trainIndex <- createDataPartition(wdbc.data$Diagnosis, p = 0.7, list = FALSE)
trainData <- wdbc.data[trainIndex, ]  # Training data
testData <- wdbc.data[-trainIndex, ]  # Test data


# Assuming trainData and testData already exist with the Diagnosis column as the target variable
# Extract only numeric columns for training and testing datasets
trainData_knn <- trainData[, sapply(trainData, is.numeric)]  # Select only numeric columns for training data
testData_knn <- testData[, sapply(testData, is.numeric)]     # Select only numeric columns for testing data

# Make sure the labels are separated (Diagnosis is categorical)
trainLabels <- trainData$Diagnosis
testLabels <- testData$Diagnosis

# Scale the numeric data (KNN is sensitive to the scale of the data)
trainData_knn_scaled <- scale(trainData_knn)
testData_knn_scaled <- scale(testData_knn)

# Train the KNN model (Set k = 5)
k <- 5
knn_predictions <- knn(train = trainData_knn_scaled, test = testData_knn_scaled, cl = trainLabels, k = k)

# Evaluate the model using a confusion matrix
knn_conf_matrix <- confusionMatrix(knn_predictions, testLabels)

# Print the confusion matrix
print(knn_conf_matrix)

# Extract evaluation metrics
accuracy_knn <- knn_conf_matrix$overall["Accuracy"]
sensitivity_knn <- knn_conf_matrix$byClass["Sensitivity"]
specificity_knn <- knn_conf_matrix$byClass["Specificity"]

# Print the evaluation metrics for KNN
cat("KNN Model Accuracy: ", accuracy_knn, "\n")
cat("KNN Model Sensitivity: ", sensitivity_knn, "\n")
cat("KNN Model Specificity: ", specificity_knn, "\n")


### tuned 

# Load necessary libraries
library(class)    # For knn
library(caret)    # For confusionMatrix

# Set seed for reproducibility
set.seed(123)

# Split the data into train and test sets
trainIndex <- createDataPartition(wdbc.data$Diagnosis, p = 0.7, list = FALSE)
trainData <- wdbc.data[trainIndex, ]
testData <- wdbc.data[-trainIndex, ]

# Extract only numeric columns for training and testing datasets
trainData_knn <- trainData[, sapply(trainData, is.numeric)]
testData_knn <- testData[, sapply(testData, is.numeric)]

# Separate the labels
trainLabels <- trainData$Diagnosis
testLabels <- testData$Diagnosis

# Scale the numeric data
trainData_knn_scaled <- scale(trainData_knn)
testData_knn_scaled <- scale(testData_knn)

# Define a range of k values to test
k_values <- seq(1, 20, by = 2)  # Test odd values of k from 1 to 20

# Initialize a variable to store the results
cv_results <- data.frame(k = k_values, Accuracy = NA)

# Perform cross-validation for each k
for (i in 1:length(k_values)) {
  k <- k_values[i]
  
  # Use cross-validation on training data to compute accuracy
  knn_cv <- train(
    x = trainData_knn_scaled,
    y = trainLabels,
    method = "knn",
    tuneGrid = data.frame(k = k),
    trControl = trainControl(method = "cv", number = 10)  # 10-fold cross-validation
  )
  
  # Store the accuracy
  cv_results$Accuracy[i] <- max(knn_cv$results$Accuracy)
}

# Find the optimal k
best_k <- cv_results$k[which.max(cv_results$Accuracy)]
cat("Optimal k: ", best_k, "\n")

# Train the final KNN model with the optimal k
knn_predictions_tuned <- knn(
  train = trainData_knn_scaled,
  test = testData_knn_scaled,
  cl = trainLabels,
  k = best_k
)

# Evaluate the tuned model using a confusion matrix
knn_conf_matrix_tuned <- confusionMatrix(knn_predictions_tuned, testLabels)

# Print the confusion matrix
print(knn_conf_matrix_tuned)

# Extract evaluation metrics for the tuned model
accuracy_knn_tuned <- knn_conf_matrix_tuned$overall["Accuracy"]
sensitivity_knn_tuned <- knn_conf_matrix_tuned$byClass["Sensitivity"]
specificity_knn_tuned <- knn_conf_matrix_tuned$byClass["Specificity"]

# Print the evaluation metrics for the tuned KNN
cat("Tuned KNN Model Accuracy: ", accuracy_knn_tuned, "\n")
cat("Tuned KNN Model Sensitivity: ", sensitivity_knn_tuned, "\n")
cat("Tuned KNN Model Specificity: ", specificity_knn_tuned, "\n")




############## END of BLOCK-1 ###########################################333



##########BLOCK-2: #################       Logistic Regression Model (~Mihir)    ###############################


# Load necessary libraries
library(caret)
library(e1071)  # For SVM 
library(pROC)   # For ROC and AUC
library(glmnet)  # For Lasso and Ridge regression

# Step 1: Load the dataset
file_path <- "data.csv"
data <- read.csv("data.csv", header = FALSE)

# Step 2: Rename columns (assuming 32 columns: ID, Diagnosis, and 30 Features)
colnames(data) <- c("ID", "Diagnosis", paste0("Feature_", 1:30))

# Step 3: Encode the target variable (Diagnosis: 'M' -> 1, 'B' -> 0)
data$Diagnosis <- ifelse(data$Diagnosis == "M", 1, 0)

# Step 4: Convert feature columns to numeric (important step!)
data[, 3:32] <- sapply(data[, 3:32], as.numeric)

head(data[, 3:32])  # this will give the columns and rows in the table

# Step 5: Summary statistics of the features
cat("Summary Statistics:\n")
summary(data[, 3:32])



# Step 6: Count of Benign and Malignant cases
cat("\nClass Distribution:\n")
table(data$Diagnosis)


# Step 7: Visualize class distribution
ggplot(data, aes(x = as.factor(Diagnosis), fill = as.factor(Diagnosis))) +
  geom_bar() +
  labs(title = "Class Distribution (Benign vs Malignant)",
       x = "Diagnosis (0 = Benign, 1 = Malignant)", y = "Count") +
  scale_fill_manual(values = c("steelblue", "darkred"), labels = c("Benign", "Malignant")) +
  theme_minimal()

# Step 8: Check for missing values
cat("\nMissing Values Count:\n")
print(colSums(is.na(data)))

# Step 9: Pairwise correlation heatmap of features
library(corrplot)
corr_matrix <- cor(data[, 3:32], use = "complete.obs")
corrplot(corr_matrix, method = "color", type = "upper",
         title = "Correlation Heatmap of Features", tl.cex = 0.7)


# Step 10: Boxplot of selected features for Benign vs Malignant
library(tidyverse)
selected_features <- c("Feature_1", "Feature_2", "Feature_3")  # Adjust based on EDA insights
data_long <- data %>%
  pivot_longer(cols = all_of(selected_features), names_to = "Feature", values_to = "Value")

ggplot(data_long, aes(x = as.factor(Diagnosis), y = Value, fill = as.factor(Diagnosis))) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free_y") +
  labs(title = "Feature Distribution by Diagnosis", x = "Diagnosis", y = "Feature Value") +
  scale_fill_manual(values = c("steelblue", "darkred"), labels = c("Benign", "Malignant")) +
  theme_minimal()


# Step 11: Check for outliers using boxplots for key features
ggplot(data, aes(y = Feature_1, fill = as.factor(Diagnosis))) +
  geom_boxplot() +
  labs(title = "Boxplot for Feature_1 by Diagnosis", y = "Feature_1 Value", x = "Diagnosis") +
  scale_fill_manual(values = c("steelblue", "darkred")) +
  theme_minimal()



# Step 12: Prepare data for training and testing
set.seed(123)  # For reproducibility
train_index <- createDataPartition(data$Diagnosis, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Separate predictors (X) and target (y) variables
x_train <- as.matrix(train_data[, 3:32])  # Feature columns
y_train <- train_data$Diagnosis
x_test <- as.matrix(test_data[, 3:32])    # Feature columns
y_test <- test_data$Diagnosis

# Step 13: Train Lasso Logistic Regression Model
# alpha = 1 specifies Lasso; lambda is the regularization parameter
lasso_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
lasso_model



# Optimal lambda value (minimizes cross-validated error)
optimal_lambda <- lasso_model$lambda.min
cat("Optimal Lambda for Lasso:", optimal_lambda, "\n")



# Step 14: Coefficients of the Lasso Model
lasso_coefficients <- coef(lasso_model, s = "lambda.min")
print("Coefficients of the Lasso Model (optimal lambda):")
print(lasso_coefficients)

# Step 15: Predict on the Test Set
lasso_predictions <- predict(lasso_model, newx = x_test, s = "lambda.min", type = "response")
lasso_class_predictions <- ifelse(lasso_predictions > 0.5, 1, 0)




# Step 16: Evaluate the Model using ConfusionMatrix
confusion_matrix <- confusionMatrix(
  factor(lasso_class_predictions), 
  factor(y_test), 
  positive = "1"  # Specify the positive class (e.g., 1 for malignant)
)

# Print the confusion matrix and statistics
print(confusion_matrix)



# Step 17: ROC Curve and AUC
library(pROC)
roc_curve <- roc(y_test, as.numeric(lasso_predictions))
auc_value <- auc(roc_curve)

cat("\nAUC of Lasso Logistic Regression:", auc_value, "\n")

# Plot the ROC curve
plot(roc_curve, col = "blue", main = "ROC Curve for Lasso Logistic Regression")





################################# END of Logistic regression Model or BLOCK-1 ##############################


############BLOCK-3 #################### Support Vector Machine(SVM) (~Mihir)  #################################################


# Load necessary libraries
library(e1071)      # For SVM
library(caret)      # For train-test splitting, model training
library(ggplot2)    # For visualization
library(corrplot)   # For correlation plot

# Load the dataset
data <- read.csv("data.csv", header = TRUE)

# Check the structure of the dataset
str(data)


# Convert Diagnosis column to a factor (assuming it's called 'V2')
# M -> 1 (Malignant), B -> 0 (Benign)
data$V2 <- as.factor(ifelse(data$V2 == 'M', 1, 0))

# Step 1: Visualize class distribution
ggplot(data, aes(x = as.factor(V2), fill = as.factor(V2))) +
  geom_bar() +
  labs(title = "Class Distribution (Benign vs Malignant)",
       x = "Diagnosis (0 = Benign, 1 = Malignant)", y = "Count") +
  scale_fill_manual(values = c("steelblue", "darkred"), labels = c("Benign", "Malignant")) +
  theme_minimal()

# Step 2: Check for missing values
cat("\nMissing Values Count:\n")
print(colSums(is.na(data)))

# Step 3: Pairwise correlation heatmap of features
library(corrplot)
corr_matrix <- cor(data[, -2], use = "complete.obs")  # Exclude the target column (V2)
corrplot(corr_matrix, method = "color", type = "upper",
         title = "Correlation Heatmap of Features", tl.cex = 0.7)

# Step 4: Boxplot of selected features for Benign vs Malignant
library(tidyverse)
selected_features <- c("V3", "V4", "V5")  # Adjust these feature names based on your EDA
data_long <- data %>%
  pivot_longer(cols = all_of(selected_features), names_to = "Feature", values_to = "Value")

ggplot(data_long, aes(x = as.factor(V2), y = Value, fill = as.factor(V2))) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free_y") +
  labs(title = "Feature Distribution by Diagnosis", x = "Diagnosis", y = "Feature Value") +
  scale_fill_manual(values = c("steelblue", "darkred"), labels = c("Benign", "Malignant")) +
  theme_minimal()

# Step 5: Check for outliers using boxplots for key features
ggplot(data, aes(y = V3, fill = as.factor(V2))) +
  geom_boxplot() +
  labs(title = "Boxplot for Feature V3 by Diagnosis", y = "V3 Value", x = "Diagnosis") +
  scale_fill_manual(values = c("steelblue", "darkred")) +
  theme_minimal()




# Check for missing data
if (sum(is.na(data)) > 0) {
  stop("Dataset contains missing values.")
}

# Split the data into training (70%) and test (30%) sets
set.seed(42)  # For reproducibility
trainIndex <- createDataPartition(data$V2, p = 0.7, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Standardize the feature columns using training set parameters
scaling_params <- preProcess(trainData[, -2], method = c("center", "scale"))
trainX <- predict(scaling_params, trainData[, -2])
testX <- predict(scaling_params, testData[, -2])

# Assign target variables
trainY <- trainData$V2
testY <- testData$V2

# Check class distribution
table(trainY)
table(testY)

# Train the initial SVM model with a radial basis function (RBF) kernel
svm_model <- svm(x = trainX, y = trainY, type = 'C-classification', kernel = 'radial', class.weights = c(`0` = 1, `1` = 1.5))

# Predict on the test set
predictions <- predict(svm_model, testX)

# Evaluate model performance using confusion matrix and other metrics
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(testY))
print(conf_matrix)


##### Tuning

tuneResult <- tune(
  svm, V2 ~ ., data = trainData,
  ranges = list(cost = 2^(-3:5), gamma = 2^(-5:0)),
  scale = FALSE
)

# Print the best parameters
print(tuneResult$best.parameters)

# Train the final model with the best parameters
best_svm_model <- tuneResult$best.model
print(best_svm_model)
# Predict on the test set with the tuned model
best_predictions <- predict(best_svm_model, testX)

# Evaluate the performance of the tuned model
best_conf_matrix <- confusionMatrix(as.factor(best_predictions), as.factor(testY))
print(best_conf_matrix)


################################### END of SVM or BLOCK-3 ###################################################



