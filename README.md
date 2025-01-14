# README: Electricity Price Forecasting Challenge

## Overview

This repository contains code for a machine learning challenge focused on forecasting electricity prices based on weather, energy production, and commercial data for two European countries - France and Germany. The objective is to explain daily variations in electricity futures prices using explanatory variables such as meteorological data, energy production, and consumption.

### Problem Context

Electricity futures are financial contracts that estimate the price of electricity at a future date. The challenge aims to build a model that predicts daily variations in the price of 24-hour futures contracts based on factors like:
- Daily temperature, wind strength, and rainfall data.
- Daily price variations of energy commodities like natural gas, coal, and carbon emissions.
- Daily electricity consumption, import/export dynamics, and energy production data (e.g., solar, wind, nuclear, hydro).

The target variable is the daily variation in electricity futures prices. The evaluation metric is the Spearman’s rank correlation between predicted variations and actual variations in the test dataset.

## Files in the Repository

1. **ElectricityPrice-QRT** - Contains the data and code for the challenge.
   - `X_train.csv`: Training input features, including weather, energy production, and consumption data.
   - `y_train.csv`: Training target variable (variation in electricity futures prices).
   - `X_test_final.csv`: Input data for making predictions on the test set.

2. **Code**: The Python code for training the model and making predictions.

3. **sublinreg.csv**: Sample output file containing predicted variations of electricity prices for the test dataset.

## Steps to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Electricity-Price-Forecasting.git
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset files (`X_train.csv`, `y_train.csv`, `X_test_final.csv`) in the `ElectricityPrice-QRT` folder.

4. Run the main Python script:
   ```bash
   python electricity_price_forecasting.py
   ```

   This will train the model, make predictions, and output the results in the file `sublinreg.csv`.

## Code Explanation

1. **Data Preprocessing**:
   - Missing values in the input data are handled by replacing them with the column mean.
   - The country variable is mapped to numeric values: 'DE' (Germany) is mapped to 0, and 'FR' (France) is mapped to 1.
   - The data is split into training and testing sets using `train_test_split`.
   - The features are scaled using `StandardScaler` for numerical features.

2. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) is used to reduce the dimensionality of the features to 9 components.

3. **Model**:
   - A linear regression model is used to predict the daily variations in electricity futures prices. The model is fitted on the reduced features (after PCA).
   
4. **Prediction**:
   - The trained model is used to predict the target variable for the test set. 
   - The predicted results are stored in a CSV file with two columns: `ID` and `TARGET`.

5. **Evaluation**:
   - The model's performance is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

6. **Submission**:
   - The predictions are saved to `sublinreg.csv` in the required format for submission.

## Example of the Output

The output CSV file (`sublinreg.csv`) contains the predicted variations in electricity futures prices with the following structure:

| ID   | TARGET |
|------|--------|
| 1    | 0.023  |
| 2    | -0.045 |
| 3    | 0.001  |
| ...  | ...    |

## Evaluation Metric

The solution is evaluated based on Spearman’s rank correlation between the predicted and actual target values in the test set. This metric measures how well the predicted variations correlate with the true variations, even if they are not numerically identical.

## Technologies Used

- **Python**: Version 3.x
- **Libraries**: 
  - Pandas for data manipulation.
  - NumPy for numerical operations.
  - Scikit-learn for machine learning algorithms, preprocessing, and evaluation.

## Notes

- The code uses Principal Component Analysis (PCA) to reduce the dimensionality of the feature space before applying linear regression.
- The `StandardScaler` is used to standardize the features, which helps improve model performance.
- The submission file is generated in the format required for the challenge, with an `ID` column and a `TARGET` column for the predicted variations.
