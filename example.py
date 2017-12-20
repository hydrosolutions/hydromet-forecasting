from hydromet_forecasting.forecasting import RegressionModel, Forecaster
from hydromet_forecasting.timeseries import FixedIndexTimeseriesCSV
import datetime



# ---------------- SETUP OF A FORECASTING MODEL ----------------

# Get a dict of available regression methods
print(RegressionModel.SupportedModels.list_models())

# Initialise a regression model class
reg_model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(2))

# Print default model parameters:
print("Default parameters: %s" %reg_model.default_parameters)

# Print possible parameter choices:
print("Possible parameters or range: %s" %reg_model.selectable_parameters)

# Set parameter and configure the regression model from the model class
model=reg_model.configure(parameters={'n_estimators': 1})

# Load example datasets od decadal timesteps (d) from csv
target=FixedIndexTimeseriesCSV("example_data/discharge_station.csv","d")
feature1=FixedIndexTimeseriesCSV("example_data/discharge_station.csv","d")
feature2=FixedIndexTimeseriesCSV("example_data/auxiliary_stationdata.csv","d")

# Set up Forecaster Object
FC_obj = Forecaster(model, target, [feature1, feature2], lag=0, laglength=[2,1], multimodel=True)




# ---------------- TRAINING & FORECASTING ----------------

# Train the model
FC_obj.train()

# Predict discharge.
# Featuresets must be of same type as when initialising FC_obj, but might contain less data than the ones used for training.
# Minimum requirement for the featuresets: they contain the datapoints that are necessary to forecast y at time t
# t is a datetime.date within the target period. It does not matter wether it is (2014,1,1) or (2014,1,10).
t = datetime.date(2014,1,10)
pred = FC_obj.predict(t,[feature1, feature2])
print("Forecast Example for 1.1 to 10.1.2014: %s" %pred)

# Raises an error when the dataset is not sufficient to predict y for the given date:
# pred = FC_obj.predict(datetime.date(2017,1,1),[feature1, feature2])




# ---------------- EVALUATING & ASSESMENT REPORT (UNDER DEVELOPMENT) ----------------

# Using cross-validation with default k_fold (--> maximum number (leave-one-out)) and generate and instance of the class Evaluate.
CV=FC_obj.cross_validate(k_fold='auto')

# write an evaluation report to a html file
CV.write_html(filename="test.html")

