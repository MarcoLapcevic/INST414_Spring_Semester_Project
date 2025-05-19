# INST414_Spring_Semester_Project
Utilizing data science techniques to predict student academic performance in higher education institutions.


# Predicting Student Academic Performance in Higher Education Institutions

## Project Overview

This project aims to use data science and machine learning (ML) techniques to predict student academic performance in higher education. Many students face academic challenges that result in delayed graduation or dropping out, which negatively affects both individual outcomes and institutional performance. By analyzing academic, demographic, and behavioral data, this project seeks to identify students at risk of underperforming early in their academic journey enabling proactive intervention.

## Problem Statement

In the context of higher education, dropout and academic failure are major concerns. This project addresses the need to predict student academic performance and dropout risk using historical data. Accurate predictions can help universities offer timely support, such as tutoring, advising, or counseling—ultimately improving retention and graduation rates.

## Repository Structure (CookieCutter DS)

project-name/
│
├── data/
│ ├── raw/ # Original dataset from Kaggle
│ └── processed/ # Cleaned data used for modeling
│
├── src/
│ ├── data/ # Scripts for data loading/cleaning
│ ├── features/ # Scripts for feature engineering
│ ├── models/ # Model training and evaluation
│ └── visualization/ # Scripts for plotting performance metrics
│
├── reports/
│ └── figures/ # Output plots and images
│
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py

## Data Source

- **Dataset**: [Predict students' dropout and academic success] (https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) 
- Source: Kaggle.com  
- Features include GPA, attendance, study time, tutoring, and more.

## Methodology

1. **Data Preparation**
   - Loaded and cleaned raw CSV data
   - Handled missing values, outliers, inconsistencies, and transformations

2. **Feature Engineering**
   - Included GPA, attendance, study hours, tutoring
   - Excluded potentially biased or irrelevant features like nationality and parental occupation

3. **Modeling Approaches**
   - `Random Forest`: Robust classification model using 100 estimators
   - `Binary Logistic Regression`: Chosen for interpretability and balanced performance

4. **Evaluation Metrics**
   - Accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix


## Model Performance Summary

| Model               | Accuracy | Precision (Dropout) | Recall (Dropout) | AUC-ROC |
|--------------------|----------|---------------------|------------------|---------|
| Random Forest       | 74%      | 0.48                | 0.20             | ~0.71   |
| Binary Logistic Regression | 74%      | 0.65                | 0.61             | ~0.78   |

- Final Model: **Binary Logistic Regression** – better recall for identifying at-risk students and easier to interpret compared to Binary Logistic Regression.

## Business/Social Value

- Early intervention enables personalized academic support (advising, tutoring, counseling)
- Helps institutions increase retention and revenue
- Promotes educational equity by supporting underserved student groups

## Ethical Considerations

- **Privacy**: Used only public data to avoid privacy violations
- **Bias Avoidance**: Excluded features that may reinforce stereotypes or systemic bias
- **Limitations**: Public dataset may not generalize perfectly to institutional data; class imbalance remains a challenge

## Instructions to Run this Project (VSCode Python Environment)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/MarcoLapcevic/INST414_Spring_Semester_Project.git
   cd INST414_Spring_Semester_Project

2. Creating a Virtual Environment
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install Dependencies
   pip install -r requirements.txt

4. Running the Pipeline
   # Step 1: Data Preparation
   python src/data/load_data.py

   # Step 2: Feature Engineering
   python src/features/build_features.py

   # Step 3: Model Training and Evaluation
   python src/models/train_model.py
   python src/models/evaluate_model.py

   # Step 4: Visualize Results
   python src/visualization/plot_metrics.py
