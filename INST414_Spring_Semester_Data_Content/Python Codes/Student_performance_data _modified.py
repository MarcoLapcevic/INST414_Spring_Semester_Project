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
