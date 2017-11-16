from hydromet_forecasting.forecasting import RegressionModel, Forecaster
from hydromet_forecasting.timeseries import FixedIndexTimeseriesCSV
import datetime


# Get a dict of available regression methods
print(RegressionModel.SupportedModels.list_models())

# Initialise a regression model
reg_model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(2))

# Print default model parameters:
print("Default parameters: %s" %reg_model.default_parameters)

# Print possible parameter choices:
print("Possible parameters or range: %s" %reg_model.selectable_parameters)

# Set parameter and configure the regression model
model=reg_model.configure(parameters={'n_estimators': 20})

# Load example datasets from csv
target=FixedIndexTimeseriesCSV("example_data/discharge_station.csv","d")
feature1=FixedIndexTimeseriesCSV("example_data/discharge_station.csv","d")
feature2=FixedIndexTimeseriesCSV("example_data/auxiliary_stationdata.csv","d")

# Set up Forecaster Object
FC_obj = Forecaster(model, target, [feature1, feature2], lag=0, laglength=[2,1], multimodel=True)

# Train the model
print("training...")
FC_obj.train()
print("training complete")


# Forecast the value for the decade from 2014-1-1 to 2014-1-10
# Remark: the featuressets given to predict() must contain at least the required values for the forecast, but additional
# values will be ignored.

# Predict discharge:
pred = FC_obj.predict(datetime.date(2014,1,1),[feature1, feature2])
print("Forecast Example for 1.1 to 10.1.2014: %s" %pred)

# Not enough data to predict discharge
print(FC_obj.predict(datetime.date(2014,1,11),[feature1, feature2]))


