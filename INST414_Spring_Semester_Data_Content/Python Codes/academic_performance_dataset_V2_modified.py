# Importing the necessary modules
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Importing and loading the original dataset
df1 = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/academic_performance_dataset_V2_cleaned.csv")

# Handle inconsistencies in categorical data
if df1['Gender'].dtype == object:
    df1['Gender'] = df1['Gender'].str.strip().str.lower().replace({
        'm': 'male', 'f': 'female', 'male': 'male', 'female': 'female'
    })

if df1['Prog Code'].dtype == object:
    df1['Prog Code'] = df1['Prog Code'].str.strip().str.upper()

# Handle missing values
# Fill numeric columns with mean
numeric_cols = df1.select_dtypes(include=['float64', 'int64']).columns
df1[numeric_cols] = df1[numeric_cols].fillna(df1[numeric_cols].mean())

# Fill categorical columns with mode
categorical_cols = df1.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df1[col] = df1[col].fillna(df1[col].mode().iloc[0])

# Encode categorical variables
df1['Gender'] = LabelEncoder().fit_transform(df1['Gender'])
df1['Prog Code'] = LabelEncoder().fit_transform(df1['Prog Code'])

# Handle outliers using IQR method
num_cols = ['CGPA', 'CGPA100', 'CGPA200', 'CGPA300', 'CGPA400', 'SGPA']
for col in num_cols:
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df1[col] = df1[col].clip(lower, upper)

# Normalize numerical variables
scaler = StandardScaler()
df1[num_cols] = scaler.fit_transform(df1[num_cols])

# Export the updated file
df1.to_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Python Codes/academic_performance_dataset_V2_modified_(export).csv", index=False)

# Most ideal data visualizations:
# 1. Histograms/KDE plots for each normalized numerical variable
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df1[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# 2. Countplots for categorical variables
for cat_col in ['Gender', 'Prog Code']:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=cat_col, data=df1)
    plt.title(f'Distribution of {cat_col}')
    plt.xlabel(cat_col)
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of All Variables')
plt.show()

# 4. Pairplot for overall relationships
sns.pairplot(df1[num_cols])
plt.suptitle('Pairplot of CGPA and SGPA Variables', y=1.02)
plt.show()

# 5. Boxplots for normalized numerical features
for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df1[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.grid(True)
    plt.show()