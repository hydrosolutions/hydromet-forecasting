from hydromet_forecasting.forecasting import RegressionModel, Forecaster
from hydromet_forecasting.timeseries import FixedIndexTimeseriesCSV
import datetime


# ---------------- SETUP OF A REGRESSION MODEL ----------------

# Get a dict of available regression methods
print(RegressionModel.SupportedModels.list_models())

# Initialise a regression model class
reg_model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(1))

# Print default model parameters:
print("Default parameters: %s" %reg_model.default_parameters)

# Print possible parameter choices:
print("Possible parameters or range: %s" %reg_model.selectable_parameters)

# Set parameter and configure the regression model from the model class
model=reg_model.configure()  #{'n_estimators':20}

# ---------------- LOADING TIMESERIES DATA FROM A FILE ----------------
# modes: "m","d","p","dl" --> monthly, decadal, pentadal, daily

discharge=FixedIndexTimeseriesCSV("example_data/decadal/Ala_Archa_short/Q.csv",mode="d").downsample('m')
precipitation=FixedIndexTimeseriesCSV("example_data/decadal/Ala_Archa_short/P.csv",mode="d").downsample('m')
temperature=FixedIndexTimeseriesCSV("example_data/decadal/Ala_Archa_short/T.csv",mode="d").downsample('m')

# ---------------- INITIALISING THE  FORECASTING OBJECT ----------------

FC_obj = Forecaster(model=model,y=discharge,X=[discharge,temperature,precipitation],laglength=[3,3,3],lag=0,multimodel=True)

# ---------------- OUTPUT A PERFORMANCE ASSESSMENT OF THE MODEL SETUP ----------------
PA_obj = FC_obj.evaluate(feedback_function=FC_obj.print_progress)
PA_obj.write_html("assessment_report.html")

# ---------------- TRAIN THE MODEL ----------------
FC_obj.train()

# ---------------- FORECAST ----------------
prediction=FC_obj.predict(targetdate=datetime.date(2011,6,2),X=[discharge,temperature,precipitation])
print(prediction)

