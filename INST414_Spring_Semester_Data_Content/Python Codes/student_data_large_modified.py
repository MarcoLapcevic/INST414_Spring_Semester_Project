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

# 1. Histogram/KDE for key numerical features (e.g., CGPA and Secondary School %)
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

# 4. PCA 2D Scatter Plot (Optional: color by gender if available)
if 'gender' in df1.columns:  # Ensure gender is present
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
