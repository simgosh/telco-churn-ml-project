library(readr)
library(dplyr)
library(readxl)
library(ggplot2)
library(MASS)
library(caret)
library(rpart)
library(randomForest)
library(xgboost)
library(pROC)
library(glmnet)
library(e1071)      # SVM 
library(caret)

churn_data <- read_csv("/Users/sim/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#explorarity data analsysis (eda)
head(churn_data)
nrow(churn_data)
table(churn_data$Churn)
dim(churn_data)
str(churn_data)
table(churn_data$gender)
table(churn_data$Contract)
table(churn_data$MultipleLines)

prop.table(table(churn_data$Churn))
ggplot(churn_data, aes(x = Churn)) +
  geom_bar(fill = c("pink", "darkgreen")) +
  labs(title = "Churn dist.", x = "Churn", y = "# of ppl")

ggplot(churn_data, aes(x = Churn, fill= gender)) +
  geom_bar(position="dodge") +
  labs(title = "Churn and Gender Dist.", x = "Churn", y = "Count") +
  scale_fill_manual(values = c("Female" = "pink", "Male" = "darkgreen")) +
  theme_minimal()
  
ggplot(churn_data,aes(x = Churn, y = tenure)) + #tenure means that who was stay as a customer
  geom_boxplot(fill = "lightgreen") +  
  labs(title = "Relationship of Tenure and Churn")

ggplot(churn_data, aes(x=Churn, y=MonthlyCharges)) +
  geom_boxplot(fill="darkblue") +
  labs(title = "Monthly Revenue vs Churn")

ggplot(churn_data, aes(x=Contract, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title = "Contract type and Churn", y="Ratio") +
  scale_fill_manual(values = c("steelblue", "tomato"))

ggplot(churn_data, aes(x=InternetService, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title = "Internet Service and Churn", y="Ratio")

ggplot(churn_data, aes(x=tenure, fill=Churn)) +
  geom_histogram(binwidth = 5, position="identity", alpha=0.9) +
  labs(title="Tenure vs Churn Dist.", x="Tenure", y="Customer Number") +
  scale_fill_manual(values = c("darkgrey", "purple"))

ggplot(churn_data, aes(x=PaymentMethod, fill=Churn)) +
  geom_bar(position="fill") +
  labs(title = "Payment Method and Churn", y="Ratio")

ggplot(churn_data, aes(x=MonthlyCharges, y=TotalCharges, color=Churn)) +
  geom_point(alpha=0.6) +
  labs(title = "MonthlyCharges vs TotalCharges by Churn",  x = "Monthly Charges",
       y = "Total Charges") +
  scale_color_manual(values = c("No" = "purple", "Yes" = "black")) +
  theme_minimal()

#checkout null value
library(dplyr)
class(churn_data)

colSums(is.na(churn_data))
summary(churn_data$TotalCharges)
#delete for "na" rows
churn_data <- na.omit(churn_data) 
colnames(churn_data)
churn_data <- dplyr::select(churn_data, -customerID)
colnames(churn_data)

#feature engineering
#churn_data <- churn_data %>%
#  mutate(across(where(is.character), as.factor))
#str(churn_data)

sample_index <- sample(seq_len(nrow(churn_data)), size = 0.8 * nrow(churn_data))
train_data <- churn_data[sample_index, ]
test_data <- churn_data[-sample_index, ]

train_data$Churn <- factor(train_data$Churn, levels = c("No", "Yes"))
test_data$Churn <- factor(test_data$Churn, levels = c("No", "Yes"))
tree_model1 <- rpart(Churn ~ ., data = train_data, method = "class", cp=0.016)
print(tree_model1)
tree_pred <- predict(tree_model1, test_data, type = "class")
table(tree_pred, test_data$Churn)


rf_model1 <- randomForest(Churn ~ ., data = train_data, ntree = 100)
print(rf_model1)
rf_pred <- predict(rf_model1, test_data)
table(rf_pred, test_data$Churn)
varImpPlot(rf_model1) #important features


churn_data$gender <- as.factor(churn_data$gender)
levels(churn_data$gender)
churn_data$Partner <- as.factor(churn_data$Partner)
levels(churn_data$Partner)
churn_data$Dependents <- as.factor(churn_data$Dependents)
levels(churn_data$Dependents)
churn_data$Contract <- as.factor(churn_data$Contract)
levels(churn_data$Contract)
churn_data$PaymentMethod <- as.factor(churn_data$PaymentMethod)
levels(churn_data$PaymentMethod)

churn_data$PhoneService <- as.factor(churn_data$PhoneService)
churn_data$MultipleLines <- as.factor(churn_data$MultipleLines)
churn_data$InternetService <- as.factor(churn_data$InternetService)
churn_data$OnlineSecurity <- as.factor(churn_data$OnlineSecurity)
churn_data$OnlineBackup <- as.factor(churn_data$OnlineBackup)
churn_data$DeviceProtection <- as.factor(churn_data$DeviceProtection)
churn_data$TechSupport <- as.factor(churn_data$TechSupport)
churn_data$StreamingTV <- as.factor(churn_data$StreamingTV)
churn_data$StreamingMovies <- as.factor(churn_data$StreamingMovies)
churn_data$PaperlessBilling <- as.factor(churn_data$PaperlessBilling)

# create a dataframe for chosen new selections
churn_data_selected <- churn_data[, c("gender", "SeniorCitizen", "TotalCharges", "MonthlyCharges", "tenure", "Contract", "PaymentMethod", "OnlineSecurity", "Churn")]

set.seed(123)  # reproducibility için
sample_index1 <- sample(seq_len(nrow(churn_data_selected)), size = 0.8 * nrow(churn_data_selected))
train_data1 <- churn_data_selected[sample_index1, ]
test_data1 <- churn_data_selected[-sample_index1, ]

# make a factor for target variabel
train_data1$Churn <- factor(train_data1$Churn, levels = c("No", "Yes"))
test_data1$Churn <- factor(test_data1$Churn, levels = c("No", "Yes"))

# create a model 
rf_model_new <- randomForest(Churn ~ ., data = train_data1, ntree = 200, mtry = 4)
rf_pred_new <- predict(rf_model_new, test_data1)

table(rf_pred_new, test_data1$Churn)
set.seed(123)

rf_grid <- expand.grid(mtry = c(2, 4, 6, 8))

train_control <- trainControl(method = "cv", number = 5, search = "grid")

rf_tuned <- train(
  Churn ~ ., data = train_data,
  method = "rf",
  metric = "Accuracy",
  tuneGrid = rf_grid,
  trControl = train_control,
  ntree = 100
)

print(rf_tuned)
plot(rf_tuned)
best_mtry <- rf_tuned$bestTune$mtry #mtry = 4

#predict for new customer
new_customer_selected <- data.frame(
  gender = factor("Male", levels = levels(train_data1$gender)),
  SeniorCitizen = 0,
  TotalCharges = 800,
  MonthlyCharges = 71,
  tenure = 25,
  Contract = factor("Month-to-month", levels = levels(train_data1$Contract)),
  PaymentMethod = factor("Mailed check", levels = levels(train_data1$PaymentMethod)),
  OnlineSecurity = factor("No", levels = levels(train_data1$OnlineSecurity))
)

rf_pred_new1 <- predict(rf_model_new, new_customer_selected, type = "prob")
print(rf_pred_new1)

#sapply(train_data1[, names(train_data1) != "Churn"], class)
#sapply(new_customer_selected, class)

churn_data_selected1 <- churn_data[, c("TechSupport", "TotalCharges", "MonthlyCharges", "tenure", "Contract", "PaymentMethod", "OnlineSecurity", "Churn")]
set.seed(123)  # reproducibility için
sample_index2 <- sample(seq_len(nrow(churn_data_selected1)), size = 0.8 * nrow(churn_data_selected1))
train_data2 <- churn_data_selected1[sample_index2, ]
test_data2 <- churn_data_selected1[-sample_index2, ]
train_data2$Churn <- factor(train_data2$Churn, levels = c("No", "Yes"))
test_data2$Churn <- factor(test_data2$Churn, levels = c("No", "Yes"))

# create a model 
rf_model_new1 <- randomForest(Churn ~ ., data = train_data2, ntree = 100)
rf_pred_new1 <- predict(rf_model_new1, test_data2)

#check for confusion matrix
confusion_mat <- table(test_data2$Churn, rf_pred_new1)
TP <- confusion_mat["Yes", "Yes"]
FP <- confusion_mat["No", "Yes"]
FN <- confusion_mat["Yes", "No"]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * precision * recall / (precision + recall)

print(paste("Precision:", round(precision, 3)))
print(paste("Recall:", round(recall, 3)))
print(paste("F1 Score:", round(f1_score, 3)))

accuracy <- sum(diag(confusion_mat)) / sum(confusion_mat)
print(accuracy)

#create a new customer for prediction
new_customer_selected1 <- data.frame(
  TechSupport = factor("No", levels = levels(train_data2$TechSupport)),
  TotalCharges = 1000,
  MonthlyCharges = 75,
  tenure = 25,
  Contract = factor("Month-to-month", levels = levels(train_data2$Contract)),
  PaymentMethod = factor("Mailed check", levels = levels(train_data2$PaymentMethod)),
  OnlineSecurity = factor("Yes", levels = levels(train_data2$OnlineSecurity))
)
rf_pred_new2 <- predict(rf_model_new1, new_customer_selected1, type = "prob")
print(rf_pred_new2)



###logistic regression
train_data$Churn <- as.factor(train_data$Churn)
test_data$Churn <- as.factor(test_data$Churn)
model_logit <- glm(Churn ~ ., data = train_data, family = "binomial")
pred_probs <- predict(model_logit, test_data, type = "response")
summary(model_logit) #tenure, totalcharges,PaperlessBillingYes, PaymentMethodElectronic check	, ContractOne year / Two year	<- these are important for model by Churn.

threshold <- 0.27
pred_class <- ifelse(pred_probs > threshold, "Yes", "No")
pred_class <- factor(pred_class, levels = levels(test_data$Churn))
confusion_mat <- table(test_data$Churn, pred_class)
print(confusion_mat)

accuracy <- sum(diag(confusion_mat)) / sum(confusion_mat)
print(paste("Accuracy:", round(accuracy, 4)))

TP <- confusion_mat["Yes", "Yes"]
FP <- confusion_mat["No", "Yes"]
FN <- confusion_mat["Yes", "No"]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * precision * recall / (precision + recall)

print(paste("Precision:", round(precision, 3)))
print(paste("Recall:", round(recall, 3)))
print(paste("F1 Score:", round(f1, 3)))

roc_obj <- roc(test_data$Churn, pred_probs)
plot(roc_obj)
auc(roc_obj)

best_threshold <- coords(roc_obj, "best", ret = "threshold")
print(best_threshold) #0.27

#model for selected variables
model_logit_selected <- glm(Churn ~ tenure + TotalCharges + PaperlessBilling + PaymentMethod + Contract, data = train_data, family = "binomial")
summary(model_logit_selected)
model_step <- stepAIC(model_logit_selected, direction = "both", trace = FALSE)

pred_probs_step <- predict(model_step, test_data, type = "response")
threshold <- 0.20  # ya da ROC'dan bulduğun en iyi threshold
pred_class_step <- ifelse(pred_probs_step > threshold, "Yes", "No")
pred_class_step <- factor(pred_class_step, levels = levels(test_data$Churn))

confusion_mat_step <- table(test_data$Churn, pred_class_step)
print(confusion_mat_step)

TP <- confusion_mat_step["Yes", "Yes"]
FP <- confusion_mat_step["No", "Yes"]
FN <- confusion_mat_step["Yes", "No"]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1 <- 2 * precision * recall / (precision + recall)

print(paste("Precision:", round(precision, 3)))
print(paste("Recall:", round(recall, 3)))
print(paste("F1 Score:", round(f1, 3)))

summary(model_step)

#PREDICT FOR NEW CUSTOMER (LOGISTIC REG MODEL)
train_data$Contract <- as.factor(train_data$Contract)
str(train_data$Contract) 
levels(train_data$Contract)
train_data$PaperlessBilling <- as.factor(train_data$PaperlessBilling)
train_data$PaymentMethod <- as.factor(train_data$PaymentMethod)

new_customer2 <- data.frame(
  tenure = 35,
  TotalCharges = 800,
  PaperlessBilling = factor("Yes", levels = c("No", "Yes")),
  PaymentMethod = factor("Electronic check", levels = levels(train_data$PaymentMethod)),
  Contract = factor("Month-to-month", levels = levels(train_data$Contract))
)

new_pred_prob <- predict(model_step, new_customer2, type = "response")
new_pred_class <- ifelse(new_pred_prob > threshold, "Yes", "No")
print(paste("Churn probability:", round(new_pred_prob, 3)))
print(paste("Predicted class:", new_pred_class))


####Lasso & Ridge Regression#####
churn_data_processed <- model.matrix(~ . -1, data = churn_data)
churn_data_processed <- churn_data_processed[, !colnames(churn_data_processed) %in% c("ChurnYes")] #delete churn column from feature matrix

y <- ifelse(churn_data$Churn == "Yes", 1, 0) #dependent variable
set.seed(123)
train_index <- sample(1:nrow(churn_data_processed), 0.8 * nrow(churn_data_processed))
X_train <- churn_data_processed[train_index, ]
X_test <- churn_data_processed[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]
colnames(churn_data_processed)
ridge_model <- glmnet(X_train, y_train,
                      family="binomial",
                      alpha = 0, #ridge
                      lambda = 10^seq(-3, 3, length = 100))

# trying to fing optimal lambda with Cross-Validation
cv_ridge <- cv.glmnet(X_train, y_train, 
                      alpha = 0,  #ridge
                      nfolds = 20,
                      type.measure = "class",
                      family = "binomial")
best_lambda_ridge <- cv_ridge$lambda.min
plot(cv_ridge)
print(best_lambda_ridge)
ridge_pred <- predict(ridge_model, s = best_lambda_ridge, 
                      newx = X_test, type = "response")
ridge_class <- ifelse(ridge_pred > 0.4, 1, 0)
confusionMatrix(factor(ridge_class), factor(y_test))


lasso_model <- glmnet(X_train, y_train,
                      family="binomial",
                      alpha = 1, #lasso
                      lambda = 10^seq(-3, 3, length = 100))

# trying to fing optimal lambda with Cross-Validation
cv_lasso <- cv.glmnet(X_train, y_train, 
                      alpha = 1,  #lasso
                      nfolds = 20,
                      type.measure = "class",
                      family = "binomial")

best_lambda_lasso <- cv_lasso$lambda.min
plot(cv_lasso)  # Plot CV error vs. log(lambda)
print(best_lambda_lasso)

#Predictions on test set (with threshold = 0.3)
lasso_pred <- predict(lasso_model, s = best_lambda_lasso, 
                      newx = X_test, type = "response")
lasso_class <- ifelse(lasso_pred > 0.4, 1, 0)  # Same threshold as Ridge

# Confusion matrix
confusionMatrix(factor(lasso_class), factor(y_test))
coef(lasso_model, s = best_lambda_lasso)
#try elastic model
elastic_model <- glmnet(X_train, y_train, alpha = 0.5, family = "binomial")


# Ridge ROC
ridge_roc <- roc(y_test, as.numeric(ridge_pred))
plot(ridge_roc, col = "blue", print.auc = TRUE, print.auc.y = 0.4)
# Lasso ROC (overlay)
plot(roc_curve, col = "red", add = TRUE, print.auc = TRUE, print.auc.y = 0.3)
legend("bottomright", legend = c("Ridge", "Lasso"), col = c("blue", "red"), lwd = 2)


#Support-Vector Machine
y_train <- as.factor(y_train)
y_test <- as.factor(y_test)

svm_model_linear <- svm(
  x = X_train,
  y = y_train,
  type = "C-classification", #for classifier
  kernel = "radial",
  scale = TRUE,  #otomatic scale
  cost = 1,#Regularization parameteri (default=1)
  sigma = 0.1,
  C = 1
)

svm_pred <- predict(svm_model_linear, X_test)
conf_matrix <- confusionMatrix(svm_pred, y_test)
print(conf_matrix$byClass) 

svm_model <- svm(
  x = X_train, 
  y = y_train,
  kernel = "radial",
  class.weights = c("0" = 1, "1" = 2)  # 1: Ayrılanları 2x önemli yap
)
svm_pred1 <- predict(svm_model, X_test)
svm_pred_factor <- factor(svm_pred1, levels = c("0", "1"))
y_test_factor <- factor(y_test, levels = c("0", "1"))
levels(svm_pred_factor)
levels(y_test_factor)
conf_matrix1 <- confusionMatrix(data = svm_pred_factor, reference = y_test_factor)
print(conf_matrix1$byClass) 

#feature selection 
rf_model <- randomForest(x = X_train, y = y_train)
var_imp <- importance(rf_model)
print(var_imp)
top_features <- names(sort(var_imp[, 1], decreasing = TRUE)[1:20])
print(top_features)

#Grid Search + Cross-Validation for SVM (hyperparameters)
svm_grid <- expand.grid(
  C = c(0.1, 1, 10),
  sigma = c(0.01, 0.05, 0.1)
)

train_control <- trainControl(method = "cv", number = 5)

svm_tuned <- train(
  x = X_train, y = y_train,
  method = "svmRadial",
  tuneGrid = svm_grid,
  trControl = train_control,
  metric = "Accuracy"
)

print(svm_tuned)
plot(svm_tuned)
best_C <- svm_tuned$bestTune$C
best_sigma <- svm_tuned$bestTune$sigma

#Feature selection and predict for new customer by svm
new_churn_selected <- churn_data[, c("TotalCharges", "MonthlyCharges", "tenure", "InternetService", "Contract", "PaymentMethod", "OnlineSecurity", "Churn")]
new_churn_selected$Churn <- factor(new_churn_selected$Churn, levels = c("No", "Yes"))
dummies <- dummyVars(~ ., data = new_churn_selected[, -8])
X <- predict(dummies, newdata = new_churn_selected[, -8])
y <- new_churn_selected$Churn
set.seed(123)
train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[-train_idx, ]
y_test <- y[-train_idx]
svm_model3 <- svm(x = X_train, y = y_train,
                 kernel = "radial",
                 type = "C-classification",
                 cost = 1,
                 scale = TRUE)

svm_pred1 <- predict(svm_model3, X_test)
conf_mat <- confusionMatrix(svm_pred1, y_test)
print(conf_mat$byClass)
levels(new_churn_selected$InternetService)
levels(new_churn_selected$Contract)
levels(new_churn_selected$PaymentMethod)
levels(new_churn_selected$OnlineSecurity)

new_customer4 <- data.frame(
  TotalCharges = 800,
  MonthlyCharges = 70,
  tenure = 24,
  InternetService = factor("Fiber optic", 
                           levels = c("DSL", "Fiber optic", "No")),
  Contract = factor("Month-to-month", 
                    levels = c("Month-to-month", "One year", "Two year")),
  PaymentMethod = factor("Electronic check", 
                         levels = c("Bank transfer (automatic)", "Credit card (automatic)", 
                                    "Electronic check", "Mailed check")),
  OnlineSecurity = factor("No", 
                          levels = c("No", "Yes", "No internet service"))
)

new_customer_dummy <- predict(dummies, new_customer4)
new_pred <- predict(svm_model3, new_customer_dummy)
print(new_pred) #result is YES so new customer will be leave (churn=yes)

