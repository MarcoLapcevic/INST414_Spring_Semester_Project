# 1. Import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import sys

# 2. Load the dataset
df = pd.read_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Chosen_datasets_original/modified_datasets/Feature Engineering/final_cleaned_dataset_fixed.csv")

# 3. Clean the data: drop any remaining missing values (safety)
df_cleaned = df.dropna()

# 4. Feature and target separation
if 'Target' not in df_cleaned.columns:
    print("Error: 'Target' column not found in the dataset.")
    print("Available columns are:", df_cleaned.columns.tolist())
    sys.exit(1)

X = df_cleaned.drop('Target', axis=1)
y = df_cleaned['Target']

# 5. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 8. Best model from grid search
best_model = grid_search.best_estimator_

# 9. Evaluate the model
y_pred = best_model.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. ROC Curve for Multiclass
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

classifier = OneVsRestClassifier(best_model)
y_score = classifier.fit(X_train_scaled, y_train).predict_proba(X_test_scaled)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = ['darkorange', 'green', 'blue']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 12. Export the dataframe as a .csv file
df.to_csv("/Users/marcolapcevic/Documents/Documents/University & College Information/University of Maryland, College Park/UMDCP Programs/Information Science Program/Semesters/Semester 6 - Spring Semester of 2025/INST414/Semester Project (Sprints)/Sprint 2/Datasets/Kaggle Datasets/Machine Learning Models/Binary Logistic Regression/binary_logistic_regression_hyperparameter_tuning_(export).csv", index=False)
