import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("./datasets/Churn_Modelling.csv")

# Step 1: Identify numeric columns and fill missing values in them
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Step 2: Remove duplicates
df.drop_duplicates(inplace=True)

# Step 3: Anonymize customer surnames (optional)
df['Surname'] = df['Surname'].apply(lambda x: 'Customer_' + str(hash(x) % 100000))

# Step 4: Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Male: 1, Female: 0
df['Geography'] = label_encoder.fit_transform(df['Geography'])

# Step 5: Feature scaling (for numerical variables)
scaler = StandardScaler()
numerical_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Step 6: Save cleaned data
df.to_csv('./datasets/cleaned_customer_churn.csv', index=False)

print("Data cleaning completed and saved to 'cleaned_customer_churn.csv'")