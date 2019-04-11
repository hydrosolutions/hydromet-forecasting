# -*- encoding: UTF-8 -*-

import os
import datetime
import argparse

from hydromet_forecasting.forecasting import RegressionModel, Forecaster
from hydromet_forecasting.timeseries import FixedIndexTimeseriesCSV

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '-f',
    '--frequency',
    help="Frequency",
    choices=(
        'fiveday',
        'decade',
        'monthly',
    ),
    default='decade'
)
arg_parser.add_argument('-l', '--language', help='Language', choices=('en', 'ru'), default='en')
args = arg_parser.parse_args()


# ---------------- SETUP OF A REGRESSION MODEL ----------------

# Get a dict of available regression methods
print(RegressionModel.SupportedModels.list_models())

# Initialise a regression model class
reg_model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(3))

# Print default model parameters:
print("Default parameters: %s" %reg_model.default_parameters)

# Print possible parameter choices:
print("Possible parameters or range: %s" %reg_model.selectable_parameters)

# Set parameter and configure the regression model from the model class
model=reg_model.configure()  #{'n_estimators':20}

# ---------------- LOADING TIMESERIES DATA FROM A FILE ----------------
# modes: "m","d","p","dl" --> monthly, decadal, pentadal, daily
if args.frequency == 'fiveday':
    discharge = FixedIndexTimeseriesCSV(
        "example_data/fiveday/Ala_Archa/Q.csv", mode='p', label="D")
    precipitation = FixedIndexTimeseriesCSV(
        "example_data/fiveday/Ala_Archa/P.csv", mode='p', label="P")
    temperature = FixedIndexTimeseriesCSV(
        "example_data/fiveday/Ala_Archa/T.csv", mode='p', label="T")
elif args.frequency == 'decade':
    discharge = FixedIndexTimeseriesCSV(
        "example_data/decadal/Ala_Archa/Q.csv", mode='d', label="D")
    precipitation = FixedIndexTimeseriesCSV(
        "example_data/decadal/Ala_Archa/P.csv", mode='d', label="P")
    temperature = FixedIndexTimeseriesCSV(
        "example_data/decadal/Ala_Archa/T.csv", mode='d', label="T")
elif args.frequency == 'monthly':
    discharge = FixedIndexTimeseriesCSV(
        "example_data/monthly/Talas_Kluchevka/Q.csv", mode='m', label="D")
    precipitation = FixedIndexTimeseriesCSV(
        "example_data/monthly/Talas_Kluchevka/PREC_ERA.csv", mode='m', label="P")
    temperature = FixedIndexTimeseriesCSV(
        "example_data/monthly/Talas_Kluchevka/TEMP_ERA.csv", mode='m', label="T")


# ---------------- INITIALISING THE  FORECASTING OBJECT ----------------

FC_obj = Forecaster(model=model,y=discharge,X=[discharge,temperature,precipitation],laglength=[3,3,3],lag=0,multimodel=True)

# ---------------- TRAIN & OUTPUT A PERFORMANCE ASSESSMENT OF THE MODEL SETUP ----------------
def print_progress(i, i_max):  print(str(i) + ' of ' + str(int(i_max)))
PA_obj = FC_obj.train_and_evaluate(feedback_function=print_progress)


PA_obj.write_html(
    frequency=args.frequency,
    username='User Name',
    organization='Organization',
    site_name='р.Чон-Кемин-устье',
    site_code='15149',
    filename='assessment_report_{}.html'.format(args.frequency),
    language=args.language,
)


# ---------------- FORECAST ----------------
prediction=FC_obj.predict(targetdate=datetime.date(2011,6,1),X=[discharge,temperature,precipitation])
print(prediction)

