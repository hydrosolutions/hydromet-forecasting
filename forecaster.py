import csv
import datetime
from math import floor
from os.path import basename

import enum
import pandas
from numpy import nan, array, isnan, full
from sklearn.base import clone
from sklearn.model_selection import KFold

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
    and forecasting values. It should work with all machine learning objects that know the methods fit() and predict().
    It was designed to work with the FixedIndexTimeseries class which handles
    timeseries that have annual periodicity. In that sense, FixedIndex means, that each year has the same number of
    periods and that every period takes the same position in every year, e.g. monthes or semi-monthes etc. It does
    not work for timeseries with periods of strict length and as such, might overlap New Year.
    However, if the option multimodel is set to False, it can work with arbitrary timeseries that are handled by a class
    that replicates the methods in FixedIndexDateUtil.

    Attributes:
        trainingdates: a list of datetime.date objects of the periods whcih where used for training. Is None before training
        evaluator: An Evaluator object of the current training state of the Forecaster instance. Is None before training
    """

    def __init__(self, model, y, X, laglength, lag=0, multimodel=True):
        """Initialising the Forecaster Object

            Args:
                model: A model object that knows the method fit() and predict() for
                        a targetvector y and a feature array X
                y: A FixedIndexTimeseries Instance that is the target data
                X: A list of FixedIndexTimeseries Instances that represent the feature data
                laglength: A list of integers that define the number of past periods that are used from the feature set.
                        Must have the same length as X
                lag: (int): when positive: the difference in days between forecasting date and the first day of the forecasted period
                            when negative: the difference in days between forecasting date and the first day of the period preceding the forecasted period
                            Example:
                                forecasted, decadal period is: 11.10-20.10,
                                lag=0, laglength=1: The forecast is done on 11.10. The period 1.10 to 10.10 is used as feature.
                                lag=4, laglength=2: The forecast is done on 7.10. The periods 21.9-30.9 and 11.9-20.9 is used as feature
                                lag=-3, laglength=1: The forecast is done on 4.10. The period 21.9 to 30.9  is used as feature.
                multimode: boolean. If true, a individual model is trained for each period of the year. Makes sense when the
                            timeseries have annual periodicity in order to differentiate seasonality.

            Returns:
                A Forecaster object with the methods train and predict

            Raises:
                ValueError: When the list "laglength" is of different length than the list X.
            """

        self._model = model
        self._multimodel = multimodel

        self._y = y
        self._y.timeseries.columns = ["target"]

        if type(X) is not list:
            self._X = [X]
        else:
            self._X = X

        self._X_type = [x.mode for x in self._X]
        self._lag = lag

        if type(laglength) is not list:
            self._laglength = [laglength]
        else:
            self._laglength = laglength

        if not len(self._laglength)==len(X):
            raise ValueError("The arguments laglength and X must be lists of the same length")

        if not self._multimodel:
            self._maxindex = 1
            self._model = [self._model]
        else:
            self._maxindex = self._y.maxindex
            self._model = [clone(self._model) for i in range(self._maxindex)]

        assert len(self._X) > 0, "predictor dataset must contain at least one feature"
        assert len(self._laglength) == len(self._X), "The list laglength must contain as many elements as X"

        self.trainingdates=None
        self.evaluator=None

    def _aggregate_featuredates(self, targetdate):
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
        if self._lag<0:
            targetdate=self._y.shift_date_by_period(targetdate, -1)
        targetdate= self._y.shift_date_by_period(targetdate, 0) - datetime.timedelta(self._lag)
        featuredates=[]
        for i,x in enumerate(self._X):
            x_targetdate=x.shift_date_by_period(targetdate,0)
            dates=[]
            for shift in range(0, self._laglength[i]):
                dates.append(x.shift_date_by_period(targetdate, -(1 + shift)))
            featuredates.append(dates)
        return featuredates

    def _aggregate_features(self, featuredates, X):
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

        X_values = full(sum(self._laglength), nan)
        k = 0

        for i, x in enumerate(X):
            try:
                X_values[k:k + self._laglength[i]] = x.timeseries[featuredates[i]].values
            except KeyError:
                pass
            k = k + self._laglength[i]
        return X_values

    def train(self, y=None):
        """Trains the model with X and y as training set

            Args:
                y: A FixedIndexTimeseries instance that contains the target data on which the model shall be trained.
                    Is meant to be used for cross validation.
                    Default: the complete available dataset given when the instance was initialised.

            Returns:
                None

            Raises:
                InsufficientData: is raised when there is not enough data to train the model for one complete year.
            """

        if not y:
            y=self._y

        X_list = [[] for i in range(self._maxindex)]
        y_list = [[] for i in range(self._maxindex)]
        trainingdate_list = []

        for index, y_value in y.timeseries.iteritems():
            if self._multimodel:
                annual_index = y.convert_to_annual_index(index)
            else:
                annual_index = 1

            featuredates = self._aggregate_featuredates(index)

            X_values = self._aggregate_features(featuredates, self._X)

            if not isnan(y_value) and not isnan(X_values).any():
                y_list[annual_index - 1].append(y_value)
                X_list[annual_index - 1].append(X_values)
                trainingdate_list.append(index)

        for i, item in enumerate(y_list):
            x_set = array(X_list[i])
            y_set = array(y_list[i])
            if len(y_set) > 0:
                try:
                    self._model[i].fit(x_set, y_set)
                except Exception as err:
                    print("An error occured while training the model for annual index %s. Please check the training data." %(i+1))
                    raise err
            else:
                raise self.InsufficientData("There is not enough data to train the model for the period with annualindex %s" %(i+1))

        self.trainingdates = trainingdate_list

    def predict(self, targetdate, X):
        """Returns the predicted value for y at targetdate

            Uses the trained model to predict y for the period that targetdate is member of.

            Args:
                targetdate: A datetime.date object that is member of the period for which y should be forecasted.
                X: A list of FixedIndexTimeseriesobjects of the type and order of the Forecaster.X_type attribute

            Returns:
                a float of the predicted value.

            Raises:
                ValueError: if X does not fit the type of X that the Forecaster instance was initialised with
                InsufficientData: is raised when the dataset in X does not contain enough data to predict y.
                ModelError: is raised when the model have not yet been trained but a forecast is requested.
            """
        type = [x.mode for x in X]
        if not type==self._X_type:
            raise ValueError("The input dataset X must be a list of FixedIndexTimeseries objects with type and length %s" % self._X_type)

        featuredates = self._aggregate_featuredates(targetdate)
        if self._multimodel:
            annual_index = self._y.convert_to_annual_index(targetdate)
        else:
            annual_index=1
        X_values = self._aggregate_features(featuredates, X)
        if not self.trainingdates:
            raise self.ModelError("There is no trained model to be used for a prediciton. Call class method .train() first.")
        elif isnan(X_values).any():
            raise self.InsufficientData("The data in X is insufficient to predict y for %s" %targetdate)
        else:
            return self._model[annual_index - 1].predict(X_values.reshape(1, -1))

    def _predict_on_trainingset(self):
        target=pandas.Series(index=self.trainingdates)
        for date in target.index:
            target[date]=self.predict(date, self._X)
        return FixedIndexTimeseries(target, mode=self._y)

    def _cross_validate(self, k_fold=5):
        # UNDER DEVELOPMENT
        y=[]

        # Aggregate data into groups for each annualindex
        if self._multimodel:
            for i in range(0, self._maxindex):
                y.append(self._y.data_by_index(i + 1))
        else:
            y.append(self._y.timeseries)


        # Split each group with KFold into training and test sets (mixes annual index again, but with equal split )
        train=[pandas.Series()] * 5
        test=[pandas.Series()] * 5
        kf = KFold(n_splits=5)
        for i, values in enumerate(y):
            k=0
            if len(y[i])>1:
                for train_index, test_index in kf.split(y[i]):
                    train[k]=train[k].append(y[i][train_index])
                    test[k]=test[k].append(y[i][test_index])
                    k+=1

        # For each KFold: train a Forecaster Object and predict the train set.
        predictions=[]
        dates=[]
        for i, trainingset in enumerate(train):
            fc=Forecaster(self._model[0], FixedIndexTimeseries(trainingset, mode=self._y.mode), self._X, self._laglength, self._lag, self._multimodel)
            fc.train()
            for target in test[i].iteritems():
                try:
                    predictions.append(fc.predict(target[0], self._X))
                    dates.append(target[0])
                    print(fc.predict(target[0], self._X))
                except:
                    pass
        predicted_ts=FixedIndexTimeseries(pandas.Series(data=predictions,index=dates).sort_index(), mode=self._y.mode)
        targeted_ts=FixedIndexTimeseries(self._y.timeseries[dates])
        return Evaluator(targeted_ts,predicted_ts)

    def trainingdata_count(self, dim=0):
        year_min = self.trainingdates[0].year
        year_max = self.trainingdates[-1].year
        mat = pandas.DataFrame(full((self._y.maxindex, year_max - year_min + 1), False, dtype=bool),
                               columns=range(year_min, year_max + 1))
        mat.index = range(1, self._y.maxindex + 1)

        for date in self.trainingdates:
            mat.loc[self._y.convert_to_annual_index(date), date.year] = True

        if dim == 0:
            return mat.sum().sum()
        elif dim==1:
            return mat.sum(axis=1)
        elif dim==2:
            return mat

    class InsufficientData(Exception):
        pass

    class ModelError(Exception):
        pass

class FixedIndexTimeseries(object):
    """This class implements a wrapper for 5-day, decadal and monthly timeseries .

    FixedIndex means, that each year has the same number of periods and that every period takes the same position in
    every year, e.g. monthes or semi-monthes etc. It does not work for timeseries with periods, that strictly consist
    of the same  number of days and as such, might overlap New Year. This class is based on pandas.Series objects.

    The timeseries are indexed by the first day of a period, e.g. 2007/5/11 for the 2nd decade in May 2007.
    The annual index is defined as the position of the period within the year, e.g. 5 for the 2nd decade in February
    Timeseries can be loaded from a csv file with the subclass FixedIndexTimeseriesCSV

    Attributes:
        timeseries: a pandas.Series object with data indexed by the first day of a period as datetime.date object.
        label: an optional, custom label for the object.
        mode: The frequency mode of the timeseries. Either p (5-day), d (decadal), m (monthly)
        maxindex: the maximum value that annualindex can have for the mode
    """

    def __init__(self, series, mode, label=None):
        """Initialises an instance of the FixedIndexTimeseries Class

            Args:
                series: A pandas.Series object, where the index is datetime.date objects.
                model: The frequency that series is expected to have, either: p (5-day), d (decadal), m (monthly)
                label: An optional label for the timeseries. Default is None: uses the label that is found in the series object.

            Returns:
                An instance of FixedIndexTimeseries

            Raises:
                ValueError: When the argument given for mode is not recognized.
                ModeError: when the mode given as argument and the property of series do not fit. Does only recognize if
                        series if of higher frequency than indicated by mode.
            """
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

        if self._check_timeseries(series):
            self.timeseries = series
        else:
            raise self.ModeError("The given series can not be recognized as a timeseries with frequency mode %s" %self.mode)

        if label==None:
            self.label=self.timeseries.name
        else:
            self.label=label

    class ModeError(Exception):
        pass

    def _check_timeseries(self,series):
        for i, item in series.iteritems():
            date=self.firstday_of_period(i.year, self.convert_to_annual_index(i))
            if not date==i:
                return False
        return True

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
            raise ValueError("Annual index is not valid: 0 < index < %s for mode=%s" % (self.maxindex + 1, self.mode))

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
        return (date.month - 1) * (self.maxindex / 12) + ((date.day - 1) / self.period) + 1

    def shift_date_by_period(self, date, shift):
        """Shifts a datetime.date object by the given number of periods.

            E.g. decadal: Shifting datetime.date(2007,1,25)
                          by -3 gives datetime.date(2006,12,21)
            Remark: The input date is fist converter to the first day of the period it is member of.

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

    def norm(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        norm=[]
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange= range(1, self.maxindex + 1)

        for index in indexrange:
            dates=map(self.firstday_of_period,years,len(years)*[index])
            norm.append(self.timeseries[dates].mean())
        if type(annualindex)==int:
            norm=norm[0]

        return norm

    def max(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            out.append(self.timeseries[dates].max())
        if type(annualindex) == int:
            out = out[0]

        return out

    def min(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            out.append(self.timeseries[dates].min())
        if type(annualindex) == int:
            out = out[0]

        return out

    def stdev_s(self, annualindex=None):
        # NEEDS TO BE SHIFTED TO EVALUATOR
        out = []
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        if annualindex:
            indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        else:
            indexrange = range(1, self.maxindex + 1)

        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            try:
                out.append(self.timeseries[dates].std())
            except:
                out.append(nan)
        if type(annualindex) == int:
            out = out[0]

        return out

    def data_by_index(self, annualindex):
        out=[]
        years = range(min(self.timeseries.index).year, max(self.timeseries.index).year + 1)
        indexrange = ([annualindex] if type(annualindex) == int else annualindex)
        for index in indexrange:
            dates = map(self.firstday_of_period, years, len(years) * [index])
            try:
                data=self.timeseries[dates]
                data=data.dropna()
                out.append(data)
            except:
                out.append([])
        if type(annualindex) == int:
            out = out[0]

        return out

class FixedIndexTimeseriesCSV(FixedIndexTimeseries):
    """Is a subclass of FixedIndexTimeseries. Can be initialised with a path of a csv file.

    Description of required csv-file format: rows contain the data of 1 year.
    The first column contains the year of each row. The length of the rows corresponds
    to number of periods of the chosen mode in each year, additional columns will be ignored
    e.g. monthly:
    1995,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
    1996,...
    Strings are loaded as NaN

    """
    def __init__(self, csv_filepath, mode, label=None):
        self.mode = mode
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
        series=self.load_csv(csv_filepath)
        FixedIndexTimeseries.__init__(self,series, mode, label)

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
                    ValueError: The yearnumber in the first column of the csv could not be recognized.
                """

            reader = csv.reader(open(filepath, 'r'))
            intlist = []
            datelist = []
            for row in reader:
                for i in range(1, self.maxindex + 1):
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

class Evaluator(object):
    """UNDER DEVELOPMENT: This class will contain all information and methods for assessing model performance

    It will have a method write_pdf(filename), that generates the assessment report and writes it to "filename".
    When no filename is given, the pdf is stored in a temporary folder.
    Returns: the pathname where the pdf is stored.
    """

    def __init__(self, y, forecast):
        self.y=y
        self.forecast=forecast

    def computeP(self):
        P=[]
        allowed_error=map(lambda x: x * 0.674, self.y.stdev_s())
        years = range(min(self.y.timeseries.index).year, max(self.y.timeseries.index).year + 1)
        for index in range(0,self.y.maxindex):
            dates = map(self.y.firstday_of_period, years, len(years) * [index+1])
            try:
                error=abs(self.forecast.timeseries[dates]-self.y.timeseries[dates])
                error.dropna()
                good=sum(error<=allowed_error[index])
                P.append(float(good)/len(error.dropna()))
            except:
                P.append(nan)
        return P

    def write_pdf(self, filename):
        import matplotlib.pyplot as plt
        import tempfile
        P=self.computeP()

        f = plt.figure()
        plt.plot(P)
        f.savefig(filename, bbox_inches='tight')
        return filename

model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(1))
model = model.configure()

ts=FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/feature1.csv","d")
FixedIndexTimeseries(ts.timeseries,"m")
test = Forecaster(model, FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/feature1.csv","m"),
                  [FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/feature1.csv","m")], lag=0, laglength=[1], multimodel=True)
test.train()
print(test.predict(datetime.date(1960,7,11),[FixedIndexTimeseriesCSV("/home/jules/Desktop/Hydromet/feature1.csv","d")]))