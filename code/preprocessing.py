import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np 

# Load the dataset
df = pd.read_csv('./datasets/Churn_Modelling.csv')

# Fill missing values with median for numeric columns, except CreditScore and Geography
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 1: Keep CreditScore and Geography as they are
# No transformations needed for these columns

# Step 2: Binning Age, Balance, and Estimated Salary
# Define the bins and labels for Age
age_bins = [18, 30, 40, 50, 60, 100]  # You can adjust these ranges as per your data
age_labels = ['18-30', '31-40', '41-50', '51-60', '60+']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

# Define the bins and labels for Balance
balance_bins = [-1, 50000, 100000, 150000, 200000, np.inf]  # Define appropriate ranges
balance_labels = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K+']
df['BalanceGroup'] = pd.cut(df['Balance'], bins=balance_bins, labels=balance_labels, include_lowest=True)

# Define the bins and labels for Estimated Salary
salary_bins = [0, 50000, 100000, 150000, 200000, np.inf]  # Define appropriate ranges
salary_labels = ['0-50K', '50K-100K', '100K-150K', '150K-200K', '200K+']
df['SalaryGroup'] = pd.cut(df['EstimatedSalary'], bins=salary_bins, labels=salary_labels, include_lowest=True)

# Step 3: Drop the original columns that have been binned if they are no longer needed
df.drop(['Age', 'Balance', 'EstimatedSalary'], axis=1, inplace=True)

# Step 6: Save cleaned data
df.to_csv('./datasets/cleaned_customer_churn.csv', index=False)

print("Data cleaning completed and saved to 'cleaned_customer_churn.csv'")