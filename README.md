# Telco Customer Churn Prediction
A data science project focused on understanding and predicting telecom customer churn.
This project analyzes customer churn for a telecommunications company using various machine learning models. The goal is to predict whether a customer will churn (leave the service) based on customer data, and to understand key factors influencing churn.

## Project Overview
Dataset: Customer data including demographics, service details, and billing information.
Objective: Predict customer churn (Yes or No) using classification models.
Methods: Exploratory Data Analysis (EDA), feature engineering, and machine learning models including:
Decision Tree,
Random Forest,
Logistic Regression,
Lasso & Ridge Regression (Regularized Logistic Regression),
Support Vector Machine (SVM),
Evaluation: Model performance assessed with confusion matrices, precision, recall, F1-score, accuracy, and ROC-AUC.

## Data Exploration and Visualization
Investigated distributions and relationships between churn and variables such as gender, tenure, contract type, internet service, and payment methods.
Visualized data using bar plots, boxplots, histograms, and scatter plots to understand trends and feature importance.

## Modeling and Evaluation
Trained and tuned models with cross-validation and hyperparameter grid search.
Evaluated models on test data using accuracy, precision, recall, F1-score, and ROC curves.
Used Random Forest variable importance and regularization methods for feature selection.
Compared models and selected best-performing ones for prediction.

## Prediction for New Customers
Demonstrated how to use trained models to predict churn probability and class for new customer profiles.

## Results by used models
### Random Forest Model Results:
The model was trained on 5625 samples with 19 features.
The best mtry parameter was selected as 4.
Looking at the test set performance: The model performs well in predicting the "No" class (938 correct predictions, about 10% error rate).
However, it has a higher error rate for the "Yes" class (187 correct, 193 incorrect).
These results indicate that the model is reasonably balanced in detecting churn but could still be improved further.

### Decision Tree Model Results : 
The decision tree was trained on 5625 samples.
The tree splits mainly on the Contract and InternetService features, with tenure further splitting one branch.
On the test set, the confusion matrix is: The model correctly predicts many "No" cases but has some difficulty detecting churn ("Yes") with 136 correct vs 244 missed.
Overall, this shows the tree captures some important patterns but has room to improve recall for churned customers.

### Logistic Regression Model Results : 
The logistic regression model was fitted to predict customer churn (Churn).
Important predictors with significant effects include:
Tenure (negative coefficient): Longer tenure reduces churn likelihood.
TotalCharges (positive): Higher total charges increase churn risk.
PaperlessBilling (Yes) (positive): Customers with paperless billing are more likely to churn.
PaymentMethodElectronic check (positive): Customers paying by electronic check have a higher chance of churn.
Contract type:
One year contract reduces churn risk.
Two year contract reduces churn risk even more strongly.
Model performance on the test set with threshold 0.27:
### Accuracy: 74.63%
### Precision: 52% (proportion of predicted churners who actually churned)
### Recall: 79.2% (proportion of actual churners correctly identified)
### F1 Score: 0.628
The model shows good ability to detect churners (high recall) but moderate precision, meaning some false positives exist.

## Ridge Regression & Lasso :
Ridge ROC-AUC: 0.860
Regularization helps prevent overfitting while maintaining high discriminatory power.

Lasso ROC-AUC: 0.861
Also performs well, with the added benefit of feature selection (some coefficients shrunk to zero).

## Support Vector Machina Model :
###  SVM with Linear Kernel : 
Accuracy: ~74.2% (from confusion matrix data)
Precision: 84.5%
Recall (Sensitivity): 93.1%
Specificity: 51.1%
F1 Score: 0.89
Balanced Accuracy: 72.1%
This model was effective in identifying churners but less effective at correctly identifying non-churners, suggesting potential over-prediction of the positive class.

### SVM with RBF Kernel (Class-Weighted) :
Precision: 90.6%
Recall: 79.0%
Specificity: 76.6%
F1 Score: 0.84
Balanced Accuracy: 77.8%
Using class weighting improved the balance between sensitivity and specificity, reducing false positives while maintaining high precision.


## Some graphs are attached below from code. 

![Rplot23](https://github.com/user-attachments/assets/8876defd-ab5e-4806-811d-9dea389bfeef)

![Rplot24](https://github.com/user-attachments/assets/9261b5ab-61c7-433c-8d29-94b1db31eacf)

![Rplot22](https://github.com/user-attachments/assets/1c94d1d3-a3ec-479b-bd68-508fc6d00410)

![Rplot21](https://github.com/user-attachments/assets/7d9b5d19-5788-4e62-9730-8dfc0ce0372b)

![Rplot20](https://github.com/user-attachments/assets/da19a4c0-f359-4728-b370-316b59f7fb86)

![Rplot19](https://github.com/user-attachments/assets/ce4dc23b-6492-4b17-828d-73bc843d9ce2)

![Rplot18](https://github.com/user-attachments/assets/fec2bf39-c771-4d30-9cbc-5187680305d3)

![Rplot17](https://github.com/user-attachments/assets/16fd2a75-44a6-4286-97b6-6dbf9d110c05)

![Rplot16](https://github.com/user-attachments/assets/fd59ba77-e371-4490-a11c-b8779e5d6aa6)
