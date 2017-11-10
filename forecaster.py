import csv
import datetime
from math import floor
from os.path import basename

import enum
import pandas
from numpy import nan, array, isnan, full
from sklearn.base import clone

#TODO make lag in days working, enhance the information output for missing data --> date, type feature set (use a object approach)

class RegressionModel(object):
    """Sets up the Predictor Model from sklearn, etc.

    Workflow:
        1. RegressionModel.SupportedModels.list_models(): returns dictionary of available models as name,value pairs
        2. model=RegressionModel.build_regression_model(RegressionModel.SupportedModels(value)): imports necessary classes (sklearn etc.)
        3. model.selectable_parameters: dictionary of possible parameters as parameter_type and "list of possible value" pairs.
           model.default_parameters: dictionary of default parameters as parameter_type and default value pairs.
        4. configured_model=model.configure(parameters): returns configured model with parameters.

    Attributes:
        default_parameters: dictionary of default parameters as parameter_type and default value pairs.
        selectable_parameters: dictionary of possible parameters as parameter_type and "list of possible value" pairs.
    """

    def __init__(self, model_class, selectable_parameters, default_parameters):
        self.model_class = model_class
        self.selectable_parameters = selectable_parameters
        self.default_parameters = default_parameters

    def configure(self, parameters=None):
        """Initialises model with parameters

            check Instance.selectable_parameters for possible parameters.

            Args:
                parameters: When no parameters are given, default parameters as defined in Instance.default_parameters are used.

            Returns:
                configured model object

            Raises:
                None
                    """
        if parameters is None:
            return self.model_class(**self.default_parameters)
        else:
            return self.model_class(**parameters)

    class SupportedModels(enum.Enum):
        """Enum class for available models:

            list_models(): Returns: dictionary of available models as name,value pairs """


        linear_regression = 1
        extra_forests = 2

        @classmethod
        def list_models(self):
            out = dict()
            for model in (self):
                out[model.name] = model.value
            return out

    @classmethod
    def build_regression_model(cls, model):
        """Returns an instance of RegressionModel

            Args:
                model: An instance of RegressionModel.SupportedModels

            Returns:
                instance of RegressionModel

            Raises:
                None
                    """

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
    """Forecasting class for timeseries that can be handled/read by FixedIndexDatetil.

    This class enables the complete workflow from setting up a timeseries model, training, evaluating
    and forecasting values. It should work with all machine learning objects that know the methods fit() and predict() for
    a targetvector y and a feature array X. It was designed to work with the FixedIndexTimeseries class which handles
    timeseries that have annual periodicity. In that sense, FixedIndex means, that each year has the same number of
    periods and that every period takes the same position in every year, e.g. monthes or semi-monthes etc. It does
    not work for timeseries with periods, that strictly consist of the same  number of days and as such, might
    overlap New Year.
    However, if the option multimodel is set to False, it can work with arbitrary timeseries that are handled by a class
    that replicates the methods in FixedIndexDateUtil.

    Attributes:
        evaluation: contains the raw output of the model evaluation.
    """

    def __init__(self, model, y, X, laglength, lag=0, multimodel=True):

        self.model = model
        self.multimodel = multimodel

        self.y = y
        self.y.timeseries.columns = ["target"]

        if type(X) is not list:
            self.X = [X]
        else:
            self.X = X

        self.X_type = [x.mode for x in self.X]
        self.lag = lag

        if type(laglength) is not list:
            self.laglength = [laglength]
        else:
            self.laglength = laglength

        if not self.multimodel:
            self.maxindex = 1
            self.model = [self.model]
        else:
            self.maxindex = self.y.maxindex
            self.model = [clone(self.model) for i in range(self.maxindex)]

        assert len(self.X) > 0, "predictor dataset must contain at least one feature"
        assert len(self.laglength)==len(self.X), "The list laglength must contain as many elements as X"

        self.evaluation=None

    def aggregate_featuredates(self, targetdate):
        """Given a targetdate, returns the list of required dates from the featuresets.

            Decadal forecast, lag 0, laglength 2:
            targetdate=datetime.date(2017,8,21) --> [datetime.date(2017,8,11),datetime.date(2017,8,1)]

            Args:
                targetdate: a datetime.date that is member of the targetperiod.

            Returns:
                A list of lists with datetime.date objects in the order of the featureset.

            Raises:
                None
            """
        if self.lag<0:
            targetdate=self.y.shift_date_by_period(targetdate,-1)
        targetdate=self.y.shift_date_by_period(targetdate,0)-datetime.timedelta(self.lag)
        featuredates=[]
        for i,x in enumerate(self.X):
            dates=[]
            for shift in range(0, self.laglength[i]):
                dates.append(x.shift_date_by_period(targetdate, -(1 + shift)))
            featuredates.append(dates)
        return featuredates

    def aggregate_features(self, featuredates, X):
        """Returns a 1D array of features for all dates in featuredates and features in X.

            The output array is in the order: feature1_t-1,feature1_t-2,feature1_t-3,feature2_t-1,feature2_t-2, and so on...

            Args:
                featuredates: A list of lists with the dates for which the data from X should be extracted
                X: A list of FixedIndexTimeseriesobjects. Its length must correspond to the length of 1st-level list of featuredates.

            Returns:
                An array with feature values

            Raises:
                None
            """

        X_values = full(sum(self.laglength), nan)
        k = 0

        for i, x in enumerate(X):
            try:
                X_values[k:k + self.laglength[i]] = x.timeseries[featuredates[i]].values
            except KeyError:
                pass
            k = k + self.laglength[i]
        return X_values

    def train(self):
        """Trains the model with X and y as training set

            Args:
                None

            Returns:
                None

            Raises:
                None
            """

        X_list = [[] for i in range(self.maxindex)]
        y_list = [[] for i in range(self.maxindex)]
        trainingdate_list = []

        for index, y_value in self.y.timeseries.iteritems():
            if self.multimodel:
                annual_index = self.y.convert_to_annual_index(index)
            else:
                annual_index = 1

            featuredates = self.aggregate_featuredates(index)

            X_values = self.aggregate_features(featuredates, self.X)

            if not isnan(y_value) and not isnan(X_values).any():
                y_list[annual_index - 1].append(y_value)
                X_list[annual_index - 1].append(X_values)
                trainingdate_list.append(index)

        for i, item in enumerate(y_list):
            X = array(X_list[i])
            y = array(y_list[i])
            if len(y) > 0:
                self.model[i].fit(X, y)
                # print(X.size)
                # print(self.model[i].coef_)
                # print(len(self.model[i].coef_))

        self.trainingdates = trainingdate_list

    def predict(self, targetdate, X):
        """Returns the predicted value for y at targetdate

            Uses the trained model to predict y. Returns NaN when noch prediction could be made due to missing data in the featureset

            Args:
                targetdate: A datetime.date object that is member of the period for which y should be forecasted.
                X: A list of FixedIndexTimeseriesobjects of the type and order of the Forecaster.X_type attribute

            Returns:
                a float of the predicted value or NaN

            Raises:
                ValueError: if X does not fit the type of X that the object was initialised with
                InsufficientData: is raised when the dataset in X does not contain enough data to predict y.
            """
        type = [x.mode for x in X]
        if not type==self.X_type:
            raise ValueError("The input dataset X must be a list of FixedIndexTimeseries objects with type and length %s" %self.X_type)

        featuredates = self.aggregate_featuredates(targetdate)
        if self.multimodel:
            annual_index = self.y.convert_to_annual_index(targetdate)
        else:
            annual_index=1
        X_values = self.aggregate_features(featuredates, X)
        if isnan(X_values).any():
            raise self.InsufficientData("The data in X is insufficient to predict y for %s" %targetdate)
        else:
            return self.model[annual_index - 1].predict(X_values.reshape(1, -1))



    def trainingdate_matrix(self):
        year_min = self.trainingdates[0].year
        year_max = self.trainingdates[-1].year
        mat = pandas.DataFrame(full((self.maxindex, year_max - year_min + 1), False, dtype=bool),
                               columns=range(year_min, year_max + 1))
        mat.index = range(1, self.maxindex + 1)
        for date in self.trainingdates:
            mat.loc[self.y.convert_to_annual_index(date), date.year] = True
        return mat

    class InsufficientData(Exception):
        pass


class FixedIndexTimeseries(object):
    """This class implements a timeserieshandler for 5-day, decadal and monthly timeseries .


    FixedIndex means, that each year has the same number of periods and that every period takes the same position in every year, e.g. monthes or semi-monthes etc. It does
    not work for timeseries with periods, that strictly consist of the same  number of days and as such, might
    overlap New Year. Pandas.timeseries and datetime.date do not (yet) support decadal or 5-day frequencies.

    Such timeseries are indexed by the first day of a period, e.g. 2007/5/11 for the 2nd decade in May 2007.
    The annual index is defined as the position of the period within the year, e.g. 5 for the 2nd decade in February
    Timeseries can be loaded from a csv file with the class method "load_csv(filepath)"

    Attributes:
        timeseries: a pandas.Series object with data indexed by the first datetime.date of a period.
        label: a string that contains the filename of the csv with whcih the class was initialized.
        maxindex: The highest annual index that a period can have within a year, e.g. 36 for decadal.
        period: The average length of a period in days (integer)
    """

    def __init__(self, csv_filepath, mode):
        self.mode=mode
        if mode == 'd':
            self.maxindex = 36
            self.period = 10
        elif mode == "p":
            self.maxindex = 72
            self.period = 5
        elif mode == "m":
            self.maxindex = 12
            self.period = 30
        else:
            raise ValueError("The given mode was not recognized. Check the docstring of the class.")
        self.timeseries=self.load_csv(csv_filepath)
        self.label=basename(csv_filepath)

    def load_csv(self, filepath):
        """loads array-like timeseries data from .csv into indexed pandas series

            Description of required csv-file format: rows contain the data of 1 year.
            The first column contains the year of each row. The length of the rows corresponds
            to number of periods of the chosen mode in each year, additional columns will be ignored
            e.g. monthly:
            1995,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
            1996,...
            Strings are loaded as NaN

            Args:
                filepath: the path to a csv file

            Returns:
                pandas.Series objects

            Raises:
                IOError: An error occured accessing the filepath
                ValueError: The yearnumber in the first column of the csv could not be recognized.
            """

        reader = csv.reader(open(filepath, 'r'))
        intlist = []
        datelist = []
        for row in reader:
            for i in range(1,self.maxindex+1):
                try:
                    intlist.append(float(row[i]))
                except:
                    intlist.append(nan)
                try:
                    date = self.firstday_of_period(year=int(row[0]), annual_index=i)
                except ValueError:
                    raise ValueError("CSV format error: The first column must contain years")
                datelist.append(date)

        return pandas.Series(data=intlist, index=datelist, name=basename(filepath))



    def firstday_of_period(self, year, annual_index):
        """Returns the first day of a period given by the year and the annual index of the period

            Decadal: first day of period (2007,3) --> datetime.date(2007,1,21)

            Args:
                year: The year
                annual_index: The index of the period within a year. 0 < annual_index < maxindex (e.g. 5-day: 72)

            Returns:
                datetime.date(y,m,d) of the first day of the period described by the year and annnual index.

            Raises:
                ValueError: When the annual index is invalid or outside the valid range defined by the mode
            """
        if not 0 < annual_index < self.maxindex+1 or not type(annual_index)==int:
            raise ValueError("Annual index is not valid: 0 < index < %s for mode=%s" % (self.maxindex+1,self.mode))

        month = int((annual_index - 1) / (self.maxindex / 12)) + 1
        day_start = ((annual_index - 1) % (self.maxindex / 12)) * self.period + 1
        return datetime.date(year, month, day_start)

    def convert_to_annual_index(self, date):
        """Returns the annual_index of a datetime.date object

            Decadal: datetime.date(2007,1,21) --> first day of period (2007,3)
                     datetime.date(2007,1,30) --> first day of period (2007,3)
                     datetime.date(2007,2,1)  --> first day of period (2007,4)
            Is the reverse function of firstday_of_period(year,annual_index)

            Args:
                date: A datetime.date object

            Returns:
                int: the annual index of the period that the datetime.date is member of.

            Raises:
                None
            """
        return (date.month - 1) * (self.maxindex / 12) + (date.day / self.period) + 1

    def shift_date_by_period(self, date, shift):
        """Shifts a datetime.date object by the given number of periods.

            E.g. decadal: Shifting datetime.date(2007,1,25)
                          by -3 gives datetime.date(2006,12,21)
            Remark: The input date is reduced to the first day of the period it is member of.

            Args:
                date: A datetime.date object
                shift: An integer corresponding to the periods that the date should be shifted.
                        Negative value: back in time. Positive value: forward in time

            Returns:
                datetime.date: the shifted date

            Raises:
                None
            """
        newindex = self.convert_to_annual_index(date) + shift
        # Correcting for shifts between years:
        if newindex < 1:
            factor = int(floor((newindex - 1) / self.maxindex))
            return self.firstday_of_period(date.year + 1 * factor, newindex - self.maxindex * factor)
        elif newindex > self.maxindex:
            factor = int(floor((newindex - 1) / self.maxindex))
            return self.firstday_of_period(date.year + int(1 * factor), newindex - self.maxindex * factor)
        else:
            return self.firstday_of_period(date.year, newindex)


model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(1))
model = model.configure()
test = Forecaster(model, FixedIndexTimeseries("/home/jules/Desktop/Hydromet/feature2.csv","p"),
                  [FixedIndexTimeseries("/home/jules/Desktop/Hydromet/feature1.csv","d"), FixedIndexTimeseries("/home/jules/Desktop/Hydromet/feature2.csv","p")], lag=5, laglength=[1,1], multimodel=True)
test.train()
print(test.predict(datetime.date(2005,12,11),
                   [FixedIndexTimeseries("/home/jules/Desktop/Hydromet/feature1.csv","d"), FixedIndexTimeseries("/home/jules/Desktop/Hydromet/feature2.csv","p")]))
