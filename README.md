# Hepatitis C Disease Prediction

## Overview
This project focuses on predicting Hepatitis C using machine learning techniques. The model was trained on a dataset in CSV format, achieving an accuracy of 95%. The purpose of this project is to assist in early detection and prediction of Hepatitis C infection based on a set of input features.

## Dataset
- **Source**: The dataset used for training the model is in CSV format.
- **Features**: Various patient-related attributes such as age, gender, ALP, AST, ALT, etc., are used for prediction.
- **Target**: Hepatitis C diagnosis (Binary classification: Positive/Negative).

## Project Structure
- `hepatitis_c_prediction.py`: The main Python script for loading data, training the model, and making predictions.
- `hepatitis_c_data.csv`: The dataset used for training the model.
- `model.pkl`: (Optional) Saved machine learning model for future predictions.
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

## Usage
The script loads the dataset, preprocesses the data, trains a machine learning model, and evaluates its performance on a test set. Predictions can be made by providing new input values to the trained model.

## Model Accuracy
- The model achieved an accuracy of approximately **95%** during testing.

## Dependencies
- Python 3.6+
- Pandas
- Scikit-learn
- NumPy

## Future Improvements
- Include additional feature engineering for improving model accuracy.
- Implement a web-based interface for easier prediction.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
Special thanks to the dataset providers and the machine learning community for their resources and tools.
