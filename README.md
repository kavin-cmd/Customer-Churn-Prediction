# Customer-Churn-Prediction

## 🚀 Overview

This project builds a **machine learning pipeline to predict customer churn** using structured customer data from a banking dataset.

By identifying customers likely to leave, the system enables **data-driven retention strategies**, directly impacting business revenue and customer lifetime value.

### Key Outcomes
- Achieved **~85–90% classification accuracy** on churn prediction  
- Improved model performance by **~10–15%** through feature engineering and hyperparameter tuning  
- Identified key churn drivers such as **age, balance, and activity level**

---

## 🧠 Approach

End-to-end ML workflow:

- **Data Cleaning:** Handling missing values, encoding categorical variables, scaling features  
- **Exploratory Data Analysis (EDA):** Identifying trends and correlations  
- **Feature Engineering:** Improving signal quality for predictive modeling  
- **Modeling:** Training and evaluating classification models  
- **Optimization:** Hyperparameter tuning for performance improvement  

---

## 📊 Dataset

The dataset includes customer attributes such as:

- Demographics (Age, Gender, Geography)  
- Financial information (Balance, Credit Score, Salary)  
- Behavioral indicators (Tenure, Activity, Product usage)  

**Target Variable:**  
- `Exited` → Indicates whether a customer churned  

---

## ⚙️ Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**

---

## 🔄 Pipeline

### 1. Data Preprocessing
- Cleaned and transformed raw data  
- Encoded categorical features and scaled numerical variables  

---

### 2. Exploratory Data Analysis
- Analyzed feature distributions and correlations  
- Identified key churn indicators (e.g., inactive users, low tenure)  

---

### 3. Model Training

- Trained classification models including:
  - Random Forest  
  - Logistic Regression (baseline)  

- Evaluated models using:
  - Accuracy  
  - Precision, Recall, F1-score  
  - ROC-AUC  

---

### 4. Hyperparameter Tuning
- Optimized model parameters to improve predictive performance  
- Achieved measurable improvement over baseline models  

---

## 📈 Results

- Final model achieved **~85–90% accuracy** on test data  
- ROC-AUC demonstrated strong separation between churned and retained customers  
- Feature importance analysis revealed:
  - Higher churn among inactive users  
  - Strong correlation with account balance and tenure  

---
