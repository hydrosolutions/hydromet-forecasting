import enum

import csv
from os import path
from numpy import nan
import pandas
import datetime

def firstday_of_decade(year, decade_of_year):
    assert 0 < decade_of_year < 37, 'decade_of_year is out of range 0 < x < 37'
    month = int((decade_of_year - 1) / 3) + 1
    day_start = ((decade_of_year - 1) % 3) * 10 + 1
    return datetime.date(year, month, day_start)

class RegressionModel(object):
    """ Supports setting up the Predictor Model

        Workflow:
        1. RegressionModel.SupportedModels.list_models(): returns dictionary of available models as name,value pairs
        2. model=RegressionModel.build_regression_model(RegressionModel.SupportedModels(value)): imports necessary classes (sklearn etc.)
        3. model.selectable_parameters: dictionary of possible parameters as parameter_type and "list of possible value" pairs.
           model.default_parameters: dictionary of default parameters as parameter_type and default value pairs.
        4. configured_model=model.configure(): returns configured model with model.default_parameters as parameters

    """

    def __init__(self, model_class, selectable_parameters, default_parameters):
        self.model_class = model_class
        self.selectable_parameters = selectable_parameters
        self.default_parameters = default_parameters

    def configure(self):
        """ initialises model_class with self.default_parameters
            Returns:
                configured model object """

        return self.model_class(self.default_parameters)

    class SupportedModels(enum.Enum):
        """ Enum class for available models:

            Methods:
                list_models(): Returns: dictionary of available models as name,value pairs """

        linear_regression = 1
        extra_forests = 2

        @classmethod
        def list_models(self):
            out=dict()
            for model in (self):
                out[model.name]=model.value
            return out

    @classmethod
    def build_regression_model(cls, model):
        """ imports selected model class (e.g. from sklearn) and returns an instance of RegressionModel """

        if model == cls.SupportedModels.linear_regression:
            from sklearn import linear_model
            return cls(linear_model.LinearRegression,
                       {'fit_intercept': [True]},
                       {'fit_intercept': True})
        elif model == cls.SupportedModels.extra_forests:
            from sklearn import ensemble
            return cls(ensemble.ExtraTreesRegressor,
                       {'n_estimators': [10, 50, 250, 1000]},
                       {'n_estimators': 50})

class Forecaster(object):

    def __init__(self, model):
        self.model=model
        self.frequency=None

class Decadal_Forecaster(Forecaster):

    def __init__(self, model, y_filepath, X_filepathlist):
        Forecaster.__init__(self,model)
        self.y=self.load_csv(y_filepath)
        self.X=self.load_csv(X_filepathlist)
        self.frequency="decadal"

    @staticmethod
    def load_csv(filepath):
        if type(filepath) is not list:
            filepathlist = [filepath]
        else:
            filepathlist=filepath

        series_dict=dict()
        for file in filepathlist:
            try:
                reader = csv.reader(open(filepath, 'r'))
            except IOError:
                print "%s is not a valid file!" % filepath

            intlist = []
            datelist = []
            for row in reader:
                for idx, stringvalue in enumerate(row[1:]):
                    try:
                        intlist.append(float(stringvalue))
                    except:
                        intlist.append(nan)
                    date = firstday_of_decade(year=int(row[0]), decade_of_year=idx + 1)
                    datelist.append(date)

            series_dict[file]=pandas.Series(data=intlist, index=datelist)

        return pandas.DataFrame(series_dict)


model=RegressionModel.build_regression_model(RegressionModel.SupportedModels(1))
model.configure()
test=Decadal_Forecaster(model,"/home/jules/Desktop/Hydromet/decadal_data.csv","/home/jules/Desktop/Hydromet/decadal_data.csv")