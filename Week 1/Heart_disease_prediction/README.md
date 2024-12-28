# Heart Disease Prediction Project

## Overview
This project leverages advanced machine learning techniques to predict the likelihood of heart disease using the Heart Disease dataset from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/heart-disease-prediction-dataset). By analyzing key health indicators and optimizing predictive models, this initiative aims to deliver a robust solution for early detection and risk assessment.

## Project Outcomes

### Analysis Summary
- Pinpointed critical health metrics most strongly associated with heart disease.
- Identified demographic trends and patterns that inform risk assessment.
- Determined optimal combinations of features to maximize predictive performance.

### Detailed Model Performance

#### Logistic Regression
- **Best Hyperparameters:**
    - Regularization Parameter (C): 1
    - Solver: liblinear
- **Performance Metrics:**
    - F1 Score: 0.8048
    - Accuracy: 80.52%
    - Precision: 81.12%
    - Recall: 80.52%
- **Confusion Matrix (Predicted vs Actual):**

|                 | Predicted Negative | Predicted Positive |
|-----------------|--------------------|--------------------|
| Actual Negative | 119                | 40                 |
| Actual Positive | 20                 | 129                |

- **Model File:** `Models/LogisticRegression_model.pkl`

--------------------------------------------

#### Decision Tree Classifier
- **Best Hyperparameters:**
    - Maximum Depth: 20
    - Minimum Samples Split: 2
- **Performance Metrics:**
    - F1 Score: 0.9707
    - Accuracy: 0.9708
    - Precision: 0.9723
    - Recall: 0.9708
- **Confusion Matrix (Predicted vs Actual):**

|                 | Predicted Negative | Predicted Positive |
|-----------------|--------------------|--------------------|
| Actual Negative | 159                | 0                  |
| Actual Positive | 9                  | 140                |

- **Model File:** `Models/DecisionTreeClassifier_model.pkl`

--------------------------------------------

#### Random Forest Classifier (Best Performing)
- **Best Hyperparameters:**
    - Maximum Depth: None
    - Number of Estimators: 100
- **Performance Metrics:**
   - F1 Score: 0.9903
   - Accuracy: 0.9903
   - Precision: 0.9904
   - Recall: 0.9903
- **Confusion Matrix (Predicted vs Actual):**

|                 | Predicted Negative | Predicted Positive |
|-----------------|--------------------|--------------------|
| Actual Negative | 156                | 0                  |
| Actual Positive | 3                  | 146                |

- **Model File:** `Models/RandomForestClassifier_model.pkl`

--------------------------------------------

#### Support Vector Machine (SVM)
- **Best Hyperparameters:**
    - Regularization Parameter (C): 2
    - Kernel: rbf
- **Performance Metrics:**
   - F1 Score: 0.9318
   - Accuracy: 0.9318
   - Precision: 0.9342
   - Recall: 0.9318
- **Confusion Matrix (Predicted vs Actual):**

|                 | Predicted Negative | Predicted Positive |
|-----------------|--------------------|--------------------|
| Actual Negative | 143                | 16                 |
| Actual Positive | 5                 | 144                |

- **Model File:** `Models/SVC_model.pkl`

### Key Insights
1. The Random Forest Classifier demonstrated the best overall performance, achieving the highest accuracy and F1 score.
2. The Decision Tree Classifier exhibited near-perfect precision for negative predictions, making it a viable alternative in certain contexts.
3. While Logistic Regression and SVM produced comparable results, they were outperformed by ensemble methods due to the latter's ability to mitigate overfitting and capitalize on diverse predictive strengths.
4. All models were successfully optimized and saved for seamless deployment.

### Technical Implementation
- Developed using Python with the scikit-learn library.
- Hyperparameter optimization performed using GridSearchCV.
- Models were serialized into PKL files for ease of deployment and integration.
- A comprehensive suite of evaluation metrics was implemented to ensure rigorous model assessment.

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn
- Pickle for model serialization

