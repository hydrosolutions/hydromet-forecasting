from hydromet_forecasting.forecasting import RegressionModel, Forecaster
from hydromet_forecasting.timeseries import FixedIndexTimeseriesCSV
import datetime


# ---------------- SETUP OF A FORECASTING MODEL ----------------

# Get a dict of available regression methods
print(RegressionModel.SupportedModels.list_models())

# Initialise a regression model class
reg_model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(6))

# Print default model parameters:
print("Default parameters: %s" %reg_model.default_parameters)

# Print possible parameter choices:
print("Possible parameters or range: %s" %reg_model.selectable_parameters)

# Set parameter and configure the regression model from the model class
model=reg_model.configure()  #{'n_estimators':20}

# Load example datasets od decadal timesteps (d
Ak_Suu_15212_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ak_Suu_15212/Q.csv","d")
Ak_Suu_15212_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ak_Suu_15212/P.csv","d")
Ak_Suu_15212_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ak_Suu_15212/T.csv","d")

Ak_Tash_15256_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ak_Tash_15256/Q.csv","d")
Ak_Tash_15256_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ak_Tash_15256/P.csv","d")
Ak_Tash_15256_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ak_Tash_15256/T.csv","d")

Ala_Archa_15194_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ala_Archa_15194/Q.csv","d")
Ala_Archa_15194_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ala_Archa_15194/P.csv","d")
Ala_Archa_15194_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ala_Archa_15194/T.csv","d")

Alamedin_15189_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Alamedin_15189/Q.csv","d")
Alamedin_15189_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Alamedin_15189/P.csv","d")
Alamedin_15189_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Alamedin_15189/T.csv","d")

Besh_Tash_15283_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Besh_Tash_15283/Q.csv","d")
Besh_Tash_15283_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Besh_Tash_15283/P.csv","d")
Besh_Tash_15283_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Besh_Tash_15283/T.csv","d")

Chon_Kaindi_15216_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Chon_Kaindi_15216/Q.csv","d")
Chon_Kaindi_15216_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Chon_Kaindi_15216/P.csv","d")
Chon_Kaindi_15216_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Chon_Kaindi_15216/T.csv","d")

Chon_Kemin_15149_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Chon_Kemin_15149/Q.csv","d")
Chon_Kemin_15149_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Chon_Kemin_15149/P.csv","d")
Chon_Kemin_15149_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Chon_Kemin_15149/T.csv","d")

Kara_Balta_15215_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kara_Balta_15215/Q.csv","d")
Kara_Balta_15215_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kara_Balta_15215/P.csv","d")
Kara_Balta_15215_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kara_Balta_15215/T.csv","d")

Kegetey_15171_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kegetey_15171/Q.csv","d")
Kegetey_15171_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kegetey_15171/P.csv","d")
Kegetey_15171_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kegetey_15171/T.csv","d")

Kirov_Rodniki_15292_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kirov_Rodniki_15292/Q.csv","d")
Kirov_Rodniki_15292_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kirov_Rodniki_15292/P.csv","d")
Kirov_Rodniki_15292_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kirov_Rodniki_15292/T.csv","d")

Klyuchevka_15261_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Klyuchevka_15261/Q.csv","d")
Klyuchevka_15261_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Klyuchevka_15261/P.csv","d")
Klyuchevka_15261_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Klyuchevka_15261/T.csv","d")

Kochkor_15102_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kochkor_15102/Q.csv","d")
Kochkor_15102_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kochkor_15102/P.csv","d")
Kochkor_15102_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kochkor_15102/T.csv","d")

Kumush_Too_15287_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kumush_Too_15287/Q.csv","d")
Kumush_Too_15287_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kumush_Too_15287/P.csv","d")
Kumush_Too_15287_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Kumush_Too_15287/T.csv","d")

Sokuluk_15214_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Sokuluk_15214/Q.csv","d")
Sokuluk_15214_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Sokuluk_15214/P.csv","d")
Sokuluk_15214_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Sokuluk_15214/T.csv","d")

Ur_Maral_15285_Q=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ur_Maral_15285/Q.csv","d")
Ur_Maral_15285_P=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ur_Maral_15285/P.csv","d")
Ur_Maral_15285_T=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/hydromet_forecasting/data/decadal/Ur_Maral_15285/T.csv","d")


# Set up Forecaster Object
#FC_obj = Forecaster(model, Ak_Suu_15212_Q, [Ak_Suu_15212_Q,Ak_Suu_15212_P,Ak_Suu_15212_T,Ak_Tash_15256_P,Ak_Tash_15256_T,Ala_Archa_15194_Q,Ala_Archa_15194_P,Ala_Archa_15194_T], lag=0, laglength=[3,3,3,3,3,3,3,3], multimodel=True, decompose=False)


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

