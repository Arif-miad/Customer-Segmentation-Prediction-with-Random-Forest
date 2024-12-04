


# **Customer Segmentation Prediction with Random Forest**

This repository provides a comprehensive analysis of a customer segmentation dataset, including data preprocessing, exploratory data analysis (EDA), and a Random Forest-based prediction model. It demonstrates the complete pipeline from data cleaning to model evaluation with code examples and mathematical explanations.

---

## **Table of Contents**
1. [Dataset Description](#dataset-description)
2. [EDA (Exploratory Data Analysis)](#exploratory-data-analysis)
3. [Data Preprocessing](#data-preprocessing)
4. [Random Forest: Theory & Implementation](#random-forest-theory--implementation)
5. [Results](#results)
6. [Getting Started](#getting-started)
7. [License](#license)

---

## **Dataset Description**

The dataset contains customer information and purchasing behavior. The target is to predict customer segmentation, such as subscription status or preferred category.

### **Columns**:
1. **Customer ID**: Unique identifier for each customer.
2. **Age**: Age of the customer.
3. **Gender**: Male/Female.
4. **Purchase Amount (USD)**: Total purchase value.
5. **Category**: Product category purchased.
6. **Subscription Status**: Whether the customer has an active subscription (Yes/No).
7. Other features include payment method, shipping type, and promo code usage.

---

## **Exploratory Data Analysis**

Here, we analyze trends and patterns in the dataset to gain insights into customer behavior.

### **1. Missing Values**
```python
# Checking missing values
df.isnull().sum()
```

### **2. Distribution of Age**
```python
plt.hist(df['Age'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
```

### **3. Gender Distribution**
```python
sns.countplot(x=df['Gender'], palette='pastel')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
```

### **4. Correlation Heatmap**
```python
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```

---

## **Data Preprocessing**

Data preprocessing includes handling missing values, encoding categorical variables, and scaling numerical features.

```python
# Handling missing values
df = df.fillna(df.mode().iloc[0])

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
categorical_cols = ['Gender', 'Category', 'Payment Method']
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# Scaling numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['Age', 'Purchase Amount (USD)']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

---

## **Random Forest: Theory & Implementation**

### **Mathematical Formula**

Random Forest is an ensemble learning method that aggregates predictions from multiple decision trees.

1. **Gini Impurity**:
   \[
   G = 1 - \sum_{i=1}^{n} p_i^2
   \]
   Where \( p_i \) is the probability of class \( i \) in a node.

2. **Feature Importance**:
   \[
   FI_j = \frac{\sum_{t \in T} (I_{t}^B - I_{t}^A)}{\text{Number of trees in forest}}
   \]
   Where \( I_t^A \) and \( I_t^B \) are impurities before and after splitting on feature \( j \).

3. **Prediction**:
   \[
   \hat{y} = \text{Mode}(\{y_t\}_{t=1}^{T})
   \]
   Where \( T \) is the total number of trees.

### **Implementation**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Feature and target split
X = df.drop(columns=['Subscription Status'])
y = df['Subscription Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## **Results**

### **Performance Metrics**:
- **Accuracy**: `X%`
- **Precision**: `Y%`
- **Recall**: `Z%`

The Random Forest model achieved reliable performance in predicting customer segmentation.

---

## **Getting Started**

### **1. Clone the Repository**
```bash
git clone https://github.com/username/customer-segmentation.git
cd customer-segmentation
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Notebook**
Open the Jupyter notebook `EDA_and_Model.ipynb` and follow the analysis.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

