# COVID-19 New Cases Prediction in Malaysia

## Description
This project aims to predict the number of new COVID-19 cases in Malaysia using historical data. The code provided in this repository utilizes a Long Short-Term Memory (LSTM) neural network model to forecast the future number of new cases. The dataset used for training and testing is sourced from the Ministry of Health, Malaysia's COVID-19 public repository.

## Installation
To use this code, please follow the steps below:

1. Clone the repository to your local machine using the following command

```
git clone [repository URL]
```

2. Navigate to the project directory:

```
cd [project directory]
```

3. Install the required dependencies by running the following command:

```
pip install pandas numpy tensorflow matplotlib scikit-learn
```

4. Download the dataset from the Ministry of Health, Malaysia's COVID-19 public repository:

- Training dataset: cases_malaysia_train.csv
- Testing dataset: cases_malaysia_test.csv

  Note: URL is provided in CREDITS

5. Place the downloaded CSV files in the project directory.

## Usage
1. Open the code file in your preferred Python environment.
2. Modify the CSV_PATH_TRAIN and CSV_PATH_TEST variables in the code to point to the respective paths of the downloaded CSV files.
3. Run the code.
4. The code will perform the following steps:
- Data loading: Loads the training and testing datasets.
- Data inspection: Displays information about the datasets.
- Data cleaning: Cleans the datasets by converting data types, handling missing values, and interpolating missing values.
- Feature selection: Selects the "cases_new" column as the target variable for prediction.
- Data scaling: Scales the target variable using MinMaxScaler.
- Data windowing: Prepares the data for input into the LSTM model by creating sequences of input and output data.
- Model development: Builds an LSTM model with dropout layers.
- Model compile: Compiles the model with the Adam optimizer and mean squared error (MSE) loss function.
- Model training: Trains the model on the training data with 100 epochs.
- Model evaluation: Evaluates the model's performance by plotting loss and mean absolute percentage error (MAPE) metrics.
- Model deployment: Uses the trained model to predict new cases on the testing data and plots a graph comparing the actual and predicted new cases.

## Outputs

- loss
![epoch_loss](https://github.com/FIT003/YPAI03_NewCases_Prediction_Covid19/assets/97938451/1977faa2-ab03-404f-aab6-6e1d01a8e6a4)

- evaluation loss vs iteration
![evaluation_loss_vs_iteration](https://github.com/FIT003/YPAI03_NewCases_Prediction_Covid19/assets/97938451/5d1e1b90-9fc9-44b1-942f-1bf24a0bc942)

- actual vs predicted

![actual_vs_predicted](https://github.com/FIT003/YPAI03_NewCases_Prediction_Covid19/assets/97938451/0890c6b3-aede-469b-8272-bd4fccd49d31)

## Credit
URL: https://github.com/MoH-Malaysia/covid19-public














