# Credit Card Fraud Detection Pipeline

A machine learning-based credit card fraud detection system developed as Project 2 during my Data Science Internship at DecodeLabs.

The project focuses on detecting fraudulent financial transactions in a highly imbalanced dataset using leak-free machine learning pipelines, SMOTE, hyperparameter tuning, and a Flask-based prediction dashboard.

## Features

- Credit card transaction fraud detection
- Exploratory Data Analysis
- Imbalanced data handling using SMOTE
- Leak-free ML pipelines
- Logistic Regression and Random Forest comparison
- 5-Fold Stratified Cross-Validation
- Hyperparameter tuning using GridSearchCV
- Precision, Recall, F1 Score, and ROC-AUC evaluation
- CSV transaction dataset upload
- Fraud probability prediction
- Interactive Flask results dashboard

## Machine Learning Pipeline

Raw Transaction Data  
↓  
Stratified Train-Test Split  
↓  
SMOTE  
↓  
Model Training  
↓  
5-Fold Cross-Validation  
↓  
GridSearchCV  
↓  
Untouched Test Data Evaluation

StandardScaler was included in the Logistic Regression pipeline, while Random Forest was trained without feature scaling.

## Models Used

### Logistic Regression

- Precision: 0.0532
- Recall: 0.8737
- F1 Score: 0.1002
- ROC-AUC: 0.9634

### Random Forest

- Precision: 0.6016
- Recall: 0.8105
- F1 Score: 0.6906
- ROC-AUC: 0.9733

Random Forest was selected as the final model because it provided a significantly better balance between precision and recall.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Flask
- HTML
- CSS
- Joblib

## Project Structure

```text
Project_2_Fraud_Detection_Pipeline/
│
├── app.py
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── models/
│   └── best_fraud_detection_model.pkl
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── index.html
│   └── results.html
├── uploads/
├── requirements.txt
└── README.md