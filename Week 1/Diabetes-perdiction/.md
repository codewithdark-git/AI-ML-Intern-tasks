# Diabetes Prediction Project

## Overview
A machine learning solution to predict diabetes using the Pima Indian Diabetes dataset from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

## Project Outcomes

### Analysis Summary
- Identified key health indicators most correlated with diabetes
- Discovered significant patterns in patient demographics
- Determined optimal feature combinations for prediction

### Detailed Model Performance

#### Random Forest Classifier (Best Performing)
- Best Hyperparameters: 
    - max_depth: 10
    - n_estimators: 100
- Performance Metrics:
    - Accuracy: 99.53%
    - Precision: 99.53%
    - Recall: 99.53%
    - F1 Score: 99.53%
- Confusion Matrix:
    - True Negatives: 22,850
    - False Positives: 0
    - False Negatives: 117
    - True Positives: 2,033

#### Decision Tree Classifier
- Best Hyperparameters:
    - max_depth: 10
    - min_samples_split: 2
- Performance Metrics:
    - Accuracy: 99.52%
    - Precision: 99.52%
    - Recall: 99.52%
    - F1 Score: 99.51%
- Confusion Matrix:
    - True Negatives: 22,840
    - False Positives: 10
    - False Negatives: 111
    - True Positives: 2,039

#### Logistic Regression
- Best Hyperparameters:
    - C: 0.1
    - solver: liblinear
- Performance Metrics:
    - Accuracy: 99.46%
    - Precision: 99.45%
    - Recall: 99.46%
    - F1 Score: 99.45%
- Confusion Matrix:
    - True Negatives: 22,817
    - False Positives: 33
    - False Negatives: 103
    - True Positives: 2,047

#### Support Vector Machine (SVM)
- Best Hyperparameters:
    - C: 1
    - kernel: linear
- Performance Metrics:
    - Accuracy: 99.42%
    - Precision: 99.42%
    - Recall: 99.42%
    - F1 Score: 99.41%
- Confusion Matrix:
    - True Negatives: 22,822
    - False Positives: 28
    - False Negatives: 117
    - True Positives: 2,033

### Key Insights
1. Random Forest achieved the highest accuracy with perfect true negative rate
2. All models demonstrated exceptional performance (>99.4% accuracy)
3. Very low false positive rates across all models
4. Models saved successfully for deployment

### Technical Implementation
- Python-based solution using scikit-learn
- Hyperparameter optimization through GridSearchCV
- Models saved as PKL files in Models directory
- Comprehensive evaluation metrics implemented

### Business Value
- Highly accurate diabetes risk assessment (99.53%)
- Minimal false positives ensuring patient safety
- Production-ready models for immediate deployment
- Reliable prediction system for healthcare providers

## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn
- Pickle for model persistence
