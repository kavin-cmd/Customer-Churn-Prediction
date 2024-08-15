# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load the cleaned dataset
# df = pd.read_csv('./datasets/cleaned_customer_churn.csv')

# # Define features and target variable
# X = df[['CreditScore', 'Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'AgeGroup', 'BalanceGroup', 'SalaryGroup']]
# y = df['Exited']

# # One-hot encode categorical variables
# X_encoded = pd.get_dummies(X, columns=['Geography', 'Gender', 'AgeGroup', 'BalanceGroup', 'SalaryGroup'], drop_first=True)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # Define the RandomForest model
# model = RandomForestClassifier(random_state=42)

# # Define the parameter grid
# param_grid = {
#     # 'n_estimators': [100, 200, 300],
#     # 'max_depth': [None, 10, 20, 30],
#     # 'min_samples_split': [2, 5, 10],
#     # 'min_samples_leaf': [1, 2, 4]

# }

# # Perform GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, error_score='raise')
# grid_search.fit(X_train, y_train)

# # Best parameters
# print(f"Best Parameters: {grid_search.best_params_}")

# # Evaluate the tuned model
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Tuned Model Accuracy: {accuracy:.2f}")

# print("Hyperparameter tuning completed.") 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load the cleaned dataset
df = pd.read_csv('./datasets/cleaned_customer_churn.csv')

# Define features and target variable
X = df[['CreditScore', 'Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'AgeGroup', 'BalanceGroup', 'SalaryGroup']]
y = df['Exited']

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, columns=['Geography', 'Gender', 'AgeGroup', 'BalanceGroup', 'SalaryGroup'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Define the RandomForest model
model = RandomForestClassifier(random_state=42)

# Define the parameter grid with additional options
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Perform GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the tuned model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Tuned Model Accuracy: {accuracy:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC score
y_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")

print("Hyperparameter tuning completed.")
