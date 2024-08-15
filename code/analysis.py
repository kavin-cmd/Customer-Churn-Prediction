import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create a directory for saving plots if it does not exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the cleaned dataset
df = pd.read_csv('./datasets/cleaned_customer_churn.csv')

# Set the plot style
sns.set_theme(style="whitegrid")

# Plot 1: Correlation Matrix Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Plot 2: Histogram of Age Groups
plt.figure(figsize=(10, 6))
sns.countplot(x='AgeGroup', data=df, palette='Set2')
plt.title('Distribution of Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.savefig('plots/age_distribution.png')
plt.close()

# Plot 3: Histogram of Balance Groups
plt.figure(figsize=(10, 6))
sns.countplot(x='BalanceGroup', data=df, palette='Set2')
plt.title('Distribution of Balance Groups')
plt.xlabel('Balance Group')
plt.ylabel('Count')
plt.savefig('plots/balance_distribution.png')
plt.close()

# Plot 4: Histogram of Estimated Salary Groups
plt.figure(figsize=(10, 6))
sns.countplot(x='SalaryGroup', data=df, palette='Set2')
plt.title('Distribution of Salary Groups')
plt.xlabel('Salary Group')
plt.ylabel('Count')
plt.savefig('plots/salary_distribution.png')
plt.close()

# Plot 5: Count Plot of Geography
plt.figure(figsize=(10, 6))
sns.countplot(x='Geography', data=df, palette='Set3')
plt.title('Geography Distribution')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.savefig('plots/geography_distribution.png')
plt.close()

# Plot 6: Count Plot of Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=df, palette='Set3')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('plots/gender_distribution.png')
plt.close()

# Plot 7: Distribution of the target variable (Exited)
plt.figure(figsize=(10, 6))
sns.countplot(x='Exited', data=df, palette='Set1')
plt.title('Churn Distribution')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.savefig('plots/churn_distribution.png')
plt.close()

# Plot 8: Distribution of Credit Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['CreditScore'], kde=True, color='blue')
plt.title('Credit Score Distribution')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.savefig('plots/credit_score_distribution.png')
plt.close()

# Plot 9: Churn Rate by Geography
plt.figure(figsize=(12, 8))
sns.countplot(x='Geography', hue='Exited', data=df, palette='Set1')
plt.title('Churn Rate by Geography')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.savefig('plots/churn_rate_by_geography.png')
plt.close()

# Plot 10: Churn Rate by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='Exited', data=df, palette='Set1')
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('plots/churn_rate_by_gender.png')
plt.close()

# Summary of Binned Features
print("Summary of Binned Features:")
summary = df[['AgeGroup', 'BalanceGroup', 'SalaryGroup']].describe(include='all')
print(summary)

# Save the summary to a CSV file
summary.to_csv('plots/binned_features_summary.csv')