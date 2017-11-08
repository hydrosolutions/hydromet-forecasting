import enum
from os import path
import csv
from numpy import nan, array, isnan, full
import pandas
import datetime
from sklearn.base import clone
from math import ceil, floor

#TODO: Limit csv-reader to maxindex, X_fiellist --> not list or single item, check length not type, assert lag>-1

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

        return self.model_class(**self.default_parameters)

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

    def __init__(self, model, y_pandas_series, X_pandas_series_list, laglength, lag=0):
        self.model=model
        self.y = y_pandas_series
        self.y.columns = ["target"]
        if type(X_pandas_series_list) is not list:
            self.X = [X_pandas_series_list]
        else:
            self.X=X_pandas_series_list
        self.lag=lag
        self.laglength = laglength

        assert len(self.X) > 0, "predictor dataset must contain at least one feature"
        assert self.laglength>0, "laglength must be > 0"

    def train(self):
        pass
        #
        # df_list=[self.y]
        # for column in self.X:
        #     df_list.append(self.X[column].shift(self.lag))
        # df=pandas.concat(df_list, axis=1)
        # df=df.dropna(how='any')
        # X_list=[[] for i in range(self.maxindex)]
        # y_list = [[] for i in range(self.maxindex)]
        # model_list = [None for i in range(self.maxindex)]
        # #MUST COPY model
        # for index, row in df.iterrows():
        #     annual_index=self.convert_to_annual_decadal_index(index)
        #     y_list[annual_index-1].append(row["target"])
        #     X_list[annual_index-1].append(row[row.index!="target"])
        #
        # for i, item in enumerate(y_list):
        #     model_list[i]=self.model.fit(X_list[i],y_list[i])
        #     print(model_list[i].coef_)
        #
        # self.model.fit(X,y)

class Regular_Forecaster(Forecaster):
    '''Standard Decadal Forecast by Hydromet'''

    def __init__(self, model, y_filepath, X_filepathlist, dateutils, laglength, lag=0, multimodel=True):

        self.dateutils=dateutils
        Forecaster.__init__(self, model, self.dateutils.load_csv(y_filepath), self.dateutils.load_csv(X_filepathlist), laglength, lag)
        self.multimodel=multimodel
        if not self.multimodel:
            self.maxindex=1
            self.model=[self.model]
        else:
            self.maxindex = self.dateutils.maxindex
            self.model = [clone(self.model) for i in range(self.maxindex)]

    def aggregate_featuredates(self, targetdate):
        return [self.dateutils.shift_by_period(targetdate, -(1 + shift)) for shift in range(self.lag, self.lag + self.laglength)]

    def aggregate_features(self, featuredates, X):
        X_values = full([len(featuredates), len(X)], nan)
        for i, x in enumerate(X):
            try:
                X_values[:, i] = x[featuredates].values
            except KeyError:
                pass
        return X_values.reshape(X_values.size)

    def train(self):

        X_list=[[] for i in range(self.maxindex)]
        y_list = [[] for i in range(self.maxindex)]
        trainingdate_list=[]

        for index, y_value in self.y.iteritems():
            if self.multimodel:
                annual_index=self.dateutils.convert_to_annual_index(index)
            else:
                annual_index=1

            featuredates=self.aggregate_featuredates(index)

            X_values=self.aggregate_features(featuredates, self.X)

            if not (isnan(y_value) | isnan(X_values).any()):
                y_list[annual_index-1].append(y_value)
                X_list[annual_index-1].append(X_values)
                trainingdate_list.append(index)

        for i, item in enumerate(y_list):
            X=array(X_list[i])
            y=array(y_list[i])
            if len(y)>0:
                self.model[i].fit(X,y)
                #print(X.size)
                #print(self.model[i].coef_)
                #print(len(self.model[i].coef_))

        self.trainingdates=trainingdate_list

    def predict(self, targetdate, X_filepathlist):
        featuredates=self.aggregate_featuredates(targetdate)
        annual_index=self.dateutils.convert_to_annual_index(targetdate)
        X=self.dateutils.load_csv(X_filepathlist)
        X_values=self.aggregate_features(featuredates,X)
        return self.model[annual_index-1].predict(X_values.reshape(1,-1))

    def trainingdate_matrix(self):
        year_min=self.trainingdates[0].year
        year_max=self.trainingdates[-1].year
        mat=pandas.DataFrame(full((self.maxindex,year_max-year_min+1), False, dtype=bool), columns=range(year_min,year_max+1))
        mat.index=range(1,self.maxindex+1)
        for date in self.trainingdates:
            mat.loc[self.dateutils.convert_to_annual_index(date), date.year] = True
        return mat

class regular_dateutils(object):

    def __init__(self, mode):
        if mode=='decadal':
            self.maxindex=36
            self.period=10
        elif mode=="5-day":
            self.maxindex=72
            self.period=5
        elif mode=="monthly":
            self.maxindex=12
            self.period=30

    def load_csv(self, filepath):
        if type(filepath) is not list:
            filepathlist = [filepath]
        else:
            filepathlist=filepath

        series_list=list()
        for file in filepathlist:
            assert path.isfile(file), filepath + ' is not a file!'
            reader = csv.reader(open(file, 'r'))
            intlist = []
            datelist = []
            for row in reader:
                for idx, stringvalue in enumerate(row[1:]):
                    try:
                        intlist.append(float(stringvalue))
                    except:
                        intlist.append(nan)
                    date = self.firstday_of_period(year=int(row[0]), period_of_year=idx + 1)
                    datelist.append(date)

            series_list.append(pandas.Series(data=intlist, index=datelist))

        if type(filepath) is not list:
            return series_list[0]
        else:
            return series_list

    def firstday_of_period(self, year, period_of_year):
        assert 0 < period_of_year <= self.maxindex, 'period_of_year is out of range 0 < x < %s' % str(self.maxindex+1)
        month = int((period_of_year - 1) / (self.maxindex/12)) + 1
        day_start = ((period_of_year - 1) % (self.maxindex/12)) * self.period + 1
        return datetime.date(year, month, day_start)

    def convert_to_annual_index(self,date):
        return (date.month - 1) * (self.maxindex/12) + (date.day / self.period) + 1

    def shift_by_period(self,date, shift):
        newindex= self.convert_to_annual_index(date) + shift
        if newindex<1:
            factor=int(floor((newindex-1)/self.maxindex))
            return self.firstday_of_period(date.year + 1*factor, newindex - self.maxindex*factor)
        elif newindex>self.maxindex:
            factor = int(floor((newindex-1)/self.maxindex))
            return self.firstday_of_period(date.year + int(1*factor), newindex - self.maxindex*factor)
        else:
            return self.firstday_of_period(date.year, newindex)

model=RegressionModel.build_regression_model(RegressionModel.SupportedModels(1))
model=model.configure()
test=Regular_Forecaster(model,"/home/jules/Desktop/Hydromet/feature1.csv",["/home/jules/Desktop/Hydromet/feature1.csv"],regular_dateutils("decadal"), lag=0, laglength=1, multimodel=True )
test.train()
print(test.predict(test.dateutils.firstday_of_period(2010,1),["/home/jules/Desktop/Hydromet/feature1.csv"]))

#test=dateutils("monthly")
#date=test.firstday_of_period(2000,1)
#print(date)
#print(test.convert_to_annual_index(date))
#print(test.shift_by_period(date,-12*1000))