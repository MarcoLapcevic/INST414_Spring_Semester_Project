# Combination of prepocessed and cleaned datasets


# ============================================================================================================================
# academic_performance_dataset_V2_modified.py - dataset
# ============================================================================================================================

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
    
    
    
# ============================================================================================================================
# Student_performance_data_modified.py - dataset    
# ============================================================================================================================

# Importing the necessary modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Importing and loading the original dataset
df1 = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/Student_performance_data_cleaned.csv")

# Clean column names
df1.columns = df1.columns.str.lower().str.replace(" ", "_")

# Handle inconsistencies in categorical and numeric data
df1 = df1.drop_duplicates()

# Enforce logical bounds for certain variables
if 'gpa' in df1.columns:
    df1 = df1[(df1['gpa'] >= 0) & (df1['gpa'] <= 4)]

if 'studytimeweekly' in df1.columns:
    df1 = df1[(df1['studytimeweekly'] >= 0) & (df1['studytimeweekly'] <= 168)]

# Handle missing values
df1 = df1.dropna()

# Handle outliers using IQR method
numeric_cols = df1.select_dtypes(include=['float64', 'int64']).columns.tolist()
for col in numeric_cols:
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df1 = df1[(df1[col] >= lower) & (df1[col] <= upper)]

# Normalize numerical variables
scaler = MinMaxScaler()
df1[numeric_cols] = scaler.fit_transform(df1[numeric_cols])

# Export the updated file
df1.to_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Python Codes/Student_performance_data_modified_(export).csv", index=False)

# Most ideal data visualizations:
# 1. Violin plot for GPA distribution by Grade Class
if 'gpa' in df1.columns and 'gradeclass' in df1.columns:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='gradeclass', y='gpa', data=df1)
    plt.title('GPA Distribution by Grade Class')
    plt.grid(True)
    plt.show()

# 2. Pairplot for selected academic features
selected_cols = [col for col in ['gpa', 'studytimeweekly', 'absences'] if col in df1.columns]
if len(selected_cols) >= 2:
    sns.pairplot(df1[selected_cols])
    plt.suptitle('Pairwise Relationships Between Academic Features', y=1.02)
    plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df1[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# 4. PCA 2D scatter plot (colored by Grade Class)
if 'gradeclass' in df1.columns:
    pca = PCA(n_components=2)
    components = pca.fit_transform(df1[numeric_cols])
    df1['pc1'] = components[:, 0]
    df1['pc2'] = components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='pc1', y='pc2', hue='gradeclass', data=df1, palette='Set2')
    plt.title('PCA Plot Colored by Grade Class')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

# 5. Boxplots of selected metrics
for col in selected_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df1[col])
    plt.title(f'Boxplot of {col.upper()}')
    plt.grid(True)
    plt.show()



# ============================================================================================================================
# dataset_modified.py - dataset
# ============================================================================================================================

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

# 5. PCA 2D Scatter Plot 
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



# ============================================================================================================================
# student_data_large_modified.py - dataset
# ============================================================================================================================

# Importing the necessary modules
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Importing and loading the cleaned dataset
df1 = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/student_data_large_cleaned.csv")

# Clean column names
df1.columns = df1.columns.str.lower().str.replace(" ", "_")

# Handle missing values
# Fill numeric columns with mean
numerical_cols = df1.select_dtypes(include=['float64', 'int64']).columns
df1[numerical_cols] = df1[numerical_cols].fillna(df1[numerical_cols].mean())

# Fill categorical columns with mode
categorical_cols = df1.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df1[col] = df1[col].fillna(df1[col].mode().iloc[0])

# Handle inconsistencies in categorical data for 'gender' column, if it exists
# First, check for possible gender columns like 'gender_female' and 'gender_male'
if 'gender_female' in df1.columns or 'gender_male' in df1.columns:
    # Assuming gender is stored as binary columns 'gender_female' and 'gender_male'
    df1['gender'] = df1['gender_female'].apply(lambda x: 'Female' if x == 1 else 'Male')
elif 'gender' in df1.columns:
    # Handle case where gender column exists but inconsistencies are there
    df1['gender'] = df1['gender'].str.strip().str.lower().replace({
        'm': 'male', 'f': 'female', 'male': 'male', 'female': 'female'
    })

# Handle outliers using IQR method
num_cols = ['cgpa', 'secondary_school_percentage']
for col in num_cols:
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df1[col] = df1[col].clip(lower, upper)

# Identify numerical columns (excluding ID)
numerical_cols = df1.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'student_no.' in numerical_cols:
    numerical_cols.remove('student_no.')

# Normalize numerical variables using Min-Max scaling
scaler = MinMaxScaler()
df1[numerical_cols] = scaler.fit_transform(df1[numerical_cols])

# Export the updated file
df1.to_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Python Codes/student_data_large_modified_(export).csv", index=False)

# Most ideal data visualizations:

# 1. Histogram/KDE for key numerical features
for col in ['cgpa', 'secondary_school_percentage']:
    if col in df1.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df1[col], kde=True)
        plt.title(f'Distribution of {col.upper()}')
        plt.xlabel(col.upper())
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

# 2. Boxplots for numerical columns of interest
for col in ['cgpa', 'secondary_school_percentage']:
    if col in df1.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df1[col])
        plt.title(f'Boxplot of {col.upper()}')
        plt.xlabel(col.upper())
        plt.grid(True)
        plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(16, 12))
correlation = df1[numerical_cols].corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of All Numerical Features')
plt.show()

# 4. PCA 2D Scatter Plot
if 'gender' in df1.columns:  
    pca = PCA(n_components=2)
    components = pca.fit_transform(df1[numerical_cols])

    df1['pc1'] = components[:, 0]
    df1['pc2'] = components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='pc1', y='pc2', hue='gender', data=df1, palette='Set2')
    plt.title('2D PCA Scatter Plot Colored by Gender')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()
