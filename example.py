from hydromet_forecasting.forecasting import RegressionModel, Forecaster, SeasonalForecast
from hydromet_forecasting.timeseries import FixedIndexTimeseriesCSV
import datetime


# ---------------- SETUP OF A FORECASTING MODEL ----------------

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

# Load example datasets od decadal timesteps (d
Talas_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/monthly/Talas_Kluchevka/Q.csv","m")
Talas_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/monthly/Talas_Kluchevka/PREC_ERA.csv","m")
Talas_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/monthly/Talas_Kluchevka/TEMP_ERA.csv","m")
Talas_S = FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/daily/Talas_Kluchevka/SNOW.csv","dl")
Talas_S = Talas_S.downsample('m')

# Set up Forecaster Object
#FC_obj = Forecaster(model, Ala_Archa_15194_Q.downsample('04-09'), [Ala_Archa_15194_Q.downsample('m').detrend(),Ala_Archa_15194_P.downsample('m').detrend(),Ala_Archa_15194_T.downsample('m').detrend()], lag=0, laglength=[4,4,4], multimodel=True, decompose=False)


# ---------------- TRAINING & FORECASTING ----------------

# Train the model
#FC_obj.train()

# Predict discharge.
# Featuresets must be of same type as when initialising FC_obj, but might contain less data than the ones used for training.
# Minimum requirement for the featuresets: they contain the datapoints that are necessary to forecast y at time t
# t is a datetime.date within the target period. It does not matter wether it is (2014,1,1) or (2014,1,10).
#t = datetime.date(2014,1,1)
#pred = FC_obj.predict(t,[feature1,feature2,feature3])
#print("Forecast Example for 1.1 to 10.1.2014: %s" %pred)

# Raises an error when the dataset is not sufficient to predict y for the given date:
# pred = FC_obj.predict(datetime.date(2017,1,1),[feature1, feature2])




# ---------------- EVALUATING & ASSESMENT REPORT (UNDER DEVELOPMENT) ----------------

# Using cross-validation with default k_fold (--> maximum number (leave-one-out)) and generate and instance of the class Evaluate.
#CV=FC_obj.cross_validate(k_fold='auto')

#CV.computeP()
# write an evaluation report to a html file
#CV.y_clean.timeseries.plot()
#CV.forecast.timeseries.plot()

#CV.write_html(filename="output/test3.html")

Objs = SeasonalForecast(model=model,target=Talas_Q.downsample('04-08'),Qm=Talas_Q,Pm=Talas_P,Sm=Talas_S,Tm=Talas_T,forecast_month=4, earliest_month=2, max_features=2, n_model=30)

def print_progress(i,i_max):
    print(str(i) + ' of ' + str(int(i_max)))


Objs.train(feedback_function=print_progress)
CV = Objs.Evaluator()
CV.write_html("/home/jules/Desktop/test.html")
pred=Objs.predict(targetdate=datetime.date(2011,4,1),Qm=Talas_Q,Pm=Talas_P,Sm=Talas_S,Tm=Talas_T)
print(pred)
pass