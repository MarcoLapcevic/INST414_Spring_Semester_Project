# Importing the necessary modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Importing and loading the cleaned dataset
df1 = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/dataset_cleaned.csv")

# Handle inconsistencies in categorical data
df1.columns = df1.columns.str.lower().str.replace(" ", "_")

# Handle missing values
# Drop rows with missing values
df1 = df1.dropna()

# Handle categorical columns' inconsistencies
categorical_cols = df1.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    df1[col] = df1[col].astype(str).str.strip().str.lower()

# Handle outliers using IQR method
numerical_cols = df1.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude target variable if present
if 'target' in numerical_cols:
    numerical_cols.remove('target')

for col in numerical_cols:
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df1[col] = df1[col].clip(lower, upper)

# Normalize numerical variables using Min-Max scaling
scaler = MinMaxScaler()
df1[numerical_cols] = scaler.fit_transform(df1[numerical_cols])

# Export the updated file
df1.to_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Python Codes/dataset_modified_(export).csv", index=False)

# Most ideal data visualizations:
# 1. Countplot for the target variable
if 'target' in df1.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df1)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

# 2. Histogram/KDE for each normalized numerical column
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df1[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# 3. Boxplot for each normalized numerical column
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df1[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.grid(True)
    plt.show()

# 4. Correlation heatmap for all numerical variables
plt.figure(figsize=(12, 10))
correlation = df1.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# 5. PCA 2D Scatter Plot (if target and numerical features exist)
if 'target' in df1.columns:
    X = df1[numerical_cols]
    y = df1['target']

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    df1['pc1'] = components[:, 0]
    df1['pc2'] = components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='pc1', y='pc2', hue='target', data=df1, palette='viridis')
    plt.title('2D PCA Scatter Plot Colored by Target')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()
