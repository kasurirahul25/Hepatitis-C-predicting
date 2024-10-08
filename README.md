# Hepatitis C Disease Prediction

## Overview
This project is aimed at predicting **Hepatitis C** using machine learning techniques. The model was trained on a dataset in CSV format and achieved an accuracy of **95%**. The goal of the project is to assist in the early detection and prediction of Hepatitis C based on patient data, which can potentially lead to quicker diagnosis and treatment.

## Dataset
- **Source**: The dataset used for this project is in CSV format (`hepatitis_c_data.csv`).
- **Features**: It includes a range of patient attributes such as:
  - Age
  - Gender
  - ALP (Alkaline Phosphatase)
  - AST (Aspartate Transaminase)
  - ALT (Alanine Transaminase)
  - And other biochemical markers.
- **Target**: The target label is a binary classification (Positive/Negative) indicating whether a patient is diagnosed with Hepatitis C.

## Project Structure
- `hepatitis_c_prediction.py`: The main Python script for loading the data, preprocessing, training the model, and making predictions.
- `hepatitis_c_data.csv`: The dataset used for training and testing the model.
- `model.pkl`: The saved machine learning model for making future predictions.
- `README.md`: Project documentation (this file).

## Installation and Setup
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Hepatitis-C-Prediction.git
    cd Hepatitis-C-Prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the prediction script:
    ```bash
    python hepatitis_c_prediction.py
    ```

## Model Building and Algorithms Used
In this project, several machine learning classifiers were tested for predicting Hepatitis C. Here are the algorithms used:

1. **Logistic Regression**
2. **Random Forest Classifier** (Best performing, achieving 95% accuracy)
3. **Support Vector Machine (SVM)**
4. **K-Nearest Neighbors (KNN)**

The **Random Forest Classifier** was selected as the final model due to its superior performance in terms of accuracy and generalization.

### Code Example
Here’s a snippet of the code to load the dataset and train the model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('hepatitis_c_data.csv')

# Preprocessing
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
