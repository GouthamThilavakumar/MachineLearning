# MachineLearning
Project Overview

This project implements and evaluates multiple machine learning classification models on a single public dataset.
Each model is trained and tested using the same dataset and evaluated using standard performance metrics to ensure fair comparison.

The objective is to analyze how different classification algorithms perform on a real-world dataset.

📂 Dataset Information

Dataset Name: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository / Kaggle

Problem Type: Binary Classification

Target Classes:

0 → Malignant

1 → Benign

Dataset Size

Total Instances: 569

Total Features: 30 numerical features

Meets Assignment Constraints:

Minimum instances ≥ 500 ✅

Minimum features ≥ 12 ✅

🧠 Machine Learning Models Implemented

The following six classification models are implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

📊 Evaluation Metrics

Each model is evaluated using the following metrics:

Accuracy

AUC Score (Area Under ROC Curve)

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

MCC is particularly useful as it considers all four components of the confusion matrix and provides a balanced evaluation.

⚙️ Project Workflow

Load dataset from scikit-learn

Split dataset into training and testing sets (75% / 25%)

Apply feature scaling using StandardScaler

Train all six classification models

Evaluate models using the required metrics

Compare results in a tabular format

🛠️ Technologies & Libraries Used

Python 3.x

NumPy

Pandas

Scikit-learn

XGBoost

📦 Installation

Install the required dependencies using:

pip install numpy pandas scikit-learn xgboost

▶️ How to Run the Project

Clone or download the repository

Open the Python script or notebook

Run the file:

python classification_models.py


The evaluation metrics for all models will be printed in a tabular format

📈 Sample Output

The final output displays a comparison table containing:

Model Name

Accuracy

AUC Score

Precision

Recall

F1 Score

MCC Score

This allows easy comparison of model performance.
