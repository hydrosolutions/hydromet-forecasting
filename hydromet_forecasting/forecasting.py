import datetime

import enum
import pandas
from numpy import nan, array, isnan, full, nanmean, mean
from sklearn.base import clone
from sklearn.model_selection import KFold

from hydromet_forecasting.timeseries import FixedIndexTimeseries
from hydromet_forecasting.evaluating import Evaluator, SeasonalEvaluator

from sklearn import preprocessing
from monthdelta import monthdelta

from stldecompose import decompose as decomp
import itertools

import scipy.special as scisp

from timeit import default_timer as timer

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
        selectable_parameters: dictionary of parameters as parameter_type and a list of possible values [v1,v2,v3,v4,...]
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
                ValueError: when any parameter in parameters is invalid for this specific model.
                    """
        if parameters is None:
            return self.model_class(**self.default_parameters)
        else:
            for key in self.default_parameters:
                if not key in parameters.keys():
                    parameters.update({key: self.default_parameters[key]})
                else:
                    if not any(map(lambda x: x == parameters[key],self.selectable_parameters[key])):
                        raise ValueError("The given value for %s must be a member of the class attribte selectable parameters." %(key))

            return self.model_class(**parameters)

    class SupportedModels(enum.Enum):
        """Enum class for available models:

            list_models(): Returns: dictionary of available models as name,value pairs """

        LinearRegression = 1
        Lasso = 2
        ExtraTreesRegressor = 3


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
                ValueError: if model is not recognized
                    """

        if model == cls.SupportedModels.LinearRegression:
            from sklearn import linear_model
            return cls(linear_model.LinearRegression,
                       {'fit_intercept': [True, False]},
                       {'fit_intercept': True})
        elif model == cls.SupportedModels.Lasso:
            from sklearn import linear_model
            return cls(linear_model.Lasso,
                       {'alpha': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]},
                       {'alpha': 1})
        elif model == cls.SupportedModels.ExtraTreesRegressor:
            from sklearn import ensemble
            return cls(ensemble.ExtraTreesRegressor,
                       {'n_estimators': range(1, 41, 1),
                        'random_state': range(1,10)},
                       {'n_estimators': 10,
                        'random_state': 1})
        else:
            raise ValueError

class Forecaster(object):
    """Forecasting class for timeseries that can be handled/read by FixedIndexTimeseries.

    This class enables the complete workflow from setting up a timeseries model, training, evaluating
    and forecasting values. It should work with all machine learning objects that know the methods fit() and predict().
    It was designed to work with the FixedIndexTimeseries class which handles
    timeseries that have annual periodicity. In that sense, FixedIndex means, that each year has the same number of
    periods and that every period takes the same position in every year, e.g. monthes or semi-monthes etc. It does
    not work for timeseries with periods of strict length and as such, might overlap New Year.
    However, if the option multimodel is set to False, it can work with arbitrary timeseries that are handled by a class
    that replicates the methods in FixedIndexTimeseries.

    Attributes:
        trainingdates: a list of datetime.date objects of the periods which where used for training. Is None before training
        evaluator: An Evaluator object of the last evaluation done for this Forecaster instance. Is None before training
        trained: Boolean, False when instance has not yet been trained
    """

    def __init__(self, model, y, X, laglength, lag=0, multimodel=True, decompose=False):
        """Initialising the Forecaster Instance

            Args:
                model: A model instance that knows the method fit() and predict() for
                        a targetvector y and a feature array X
                y: A FixedIndexTimeseries Instance that is the target data
                X: A list of FixedIndexTimeseries Instances that represent the feature data
                laglength: A list of integers that define the number of past periods that are used from the feature set.
                            Must have the same length as X
                lag: (int): when negative: the difference in days between forecasting date and the first day of the forecasted period (backwards in time)
                            when positive: the difference in days between forecasting date and the first day of the period preceding the forecasted period (forward in time)
                            Example:
                                forecasted, decadal period is: 11.10-20.10,
                                lag=0, laglength=1: The forecast is done on 11.10. The period 1.10 to 10.10 is used as feature.
                                lag=-4, laglength=2: The forecast is done on 7.10. The periods 21.9-30.9 and 11.9-20.9 is used as feature
                                lag=3, laglength=1: The forecast is done on 4.10. The period 21.9 to 30.9  is used as feature.
                multimodel: boolean. If true, a individual model is trained for each period of the year. Is used to build different models
                            when the characteristics of the target timeseries have strong seasonality
                decompose: boolean: If true, the target timeseries is decomposed into seasonality and residual. The forecast is only done for the
                            residual.

            Returns:
                A Forecaster instance with the methods train, predict and cross_validate

            Raises:
                ValueError: When the list "laglength" is of different length than the list X.
            """

        self.__model = clone(model)
        self._multimodel = multimodel

        if not self._multimodel:
            self._maxindex = 1
            self.__model = [self.__model]
        else:
            self._maxindex = y.maxindex
            self.__model = [clone(self.__model) for i in range(self._maxindex)]

        self._decompose = decompose
        self._seasonal = [0 for i in range(y.maxindex)]

        self._y = y
        self._y.timeseries.columns = ["target"]

        if type(X) is not list:
            self._X = [X]
        else:
            self._X = X

        self._X_type = [x.mode for x in self._X]

        for x in self._X:
            if len(x.timeseries.dropna().index) == 0:
                raise self.__InsufficientData("the timeseries contains no data")

        self._lag = -lag # switches the sign of lag as argument, makes it easier to understand

        if type(laglength) is not list:
            self._laglength = [laglength]
        else:
            self._laglength = laglength

        if not len(self._laglength) == len(X):
            raise ValueError("The arguments laglength and X must be lists of the same length")


        self._y_scaler = [preprocessing.StandardScaler() for i in range(self._maxindex)]
        self._X_scaler = [preprocessing.StandardScaler() for i in range(self._maxindex)]

        if len(self._X) == 0:
            raise ValueError("predictor dataset must contain at least one feature")

        self.trainingdates = None
        self.evaluator = None
        self.trained = False

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
        if self._lag < 0:
            targetdate = self._y.shift_date_by_period(targetdate, -1)
        targetdate = self._y.shift_date_by_period(targetdate, 0) - datetime.timedelta(self._lag)
        featuredates = []
        for i, x in enumerate(self._X):
            x_targetdate = x.shift_date_by_period(targetdate, 0)
            dates = []
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
                ts = x.timeseries.reindex(featuredates[i]) # avoids the FutureWarning by pandas
                X_values[k:k + self._laglength[i]] = ts[featuredates[i]].values
            except KeyError:
                pass
            k = k + self._laglength[i]
        return X_values

    def train(self, y=None):
        """Trains the model with X and y as training set

            Args:
                y: A FixedIndexTimeseries instance that contains the target data on which the model shall be trained.
                    Is meant to be used for cross validation or if not all availabe data shall be used for training.
                    Default: None (the complete available dataset given when the instance was initialised is used.)

            Returns:
                None

            Raises:
                InsufficientData: is raised when there is not enough data to train the model for one complete year.
            """

        if not y:
            y = self._y

        freq = len(self._seasonal)
        if self._decompose and freq>1:
            dec = decomp(y.timeseries.values, period=freq)
            y = FixedIndexTimeseries(pandas.Series(dec.resid+dec.trend, index=y.timeseries.index), mode=y.mode)
            seasonal = FixedIndexTimeseries(pandas.Series(dec.seasonal, index=y.timeseries.index), mode=y.mode)
            self._seasonal = [nanmean(seasonal.data_by_index(i+1)) for i in range(freq)] # TODO: is self._seasonal overwritten here? Maybe choose another variable name?

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

        self.trainingdates = trainingdate_list

        for i, item in enumerate(y_list):
            if len(item) > 0:
                x_set = self._X_scaler[i].fit_transform(array(X_list[i]))
                y_set = self._y_scaler[i].fit_transform(array(y_list[i]).reshape(-1,1))
            else:
                raise self.__InsufficientData("There is not enough data to train the model for the period with annualindex %s" % (i + 1))

            try:
                self.__model[i].fit(x_set, y_set.ravel())
            except Exception as err:
                print(
                    "An error occured while training the model for annual index %s. Please check the training data." % (
                        i + 1))
                raise err
        self.trained = True

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

        if not self.trained:
            raise self.__ModelError("The model has not been trained yet.")

        type = [x.mode for x in X]
        if not type == self._X_type:
            raise ValueError(
                "The input dataset X must be a list of FixedIndexTimeseries objects with type and length %s" % self._X_type)

        featuredates = self._aggregate_featuredates(targetdate)
        X_values = self._aggregate_features(featuredates, X)

        if self._multimodel:
            annual_index = self._y.convert_to_annual_index(targetdate)
        else:
            annual_index = 1

        if isnan(X_values).any():
            raise self.__InsufficientData("The data in X is insufficient to predict y for %s" % targetdate)
        else:
            x_set = self._X_scaler[annual_index - 1].transform(X_values.reshape(1, -1))
            prediction = self.__model[annual_index - 1].predict(x_set)
            invtrans_prediction = self._y_scaler[annual_index - 1].inverse_transform(prediction.reshape(-1,1))
            return max(0,float(invtrans_prediction+self._seasonal[self._y.convert_to_annual_index(targetdate)-1]))

    def train_and_evaluate(self, k_fold='auto', feedback_function=None):
        """Conducts a crossvalidation on the Forecaster instance and returns an Evaluator instance.

            Is used to measure the performance of the forecast. Uses the scikit K-Folds cross-validator without
            shuffling.

            Args:
                k_fold: 'auto'(default) or an integer > 1, Defines in how many train/test sets the data are split. A small value
                        is better to measure real model performance but takes much longer to compute. 'auto' will choose 10 splits
                        or smaller depending on the size of the available dataset.
                feedback_function (default=None): does report on the current state of the computation. A valid feedback_function
                        must take the argument i and imax, whereby i is the current step and imax is the maximal, final step, e.g.:

                        def print_progress(i, i_max):
                            print(str(i) + ' of ' + str(int(i_max)))

            Returns:
                An instance of the Evaluator class.

            Raises:
                ValueError: if k_fold is set to a value of <1 or is not an integer.
                InsufficientData: is raised when the dataset contains too few samples for crossvalidation with
                                    the chosen value of k_fold or if it only contains one sample.
            """
        if not feedback_function:
            feedback_function = self.__no_progress

        self.train()
        trainingdate_data = FixedIndexTimeseries(self._y.timeseries.reindex(self.trainingdates), mode=self._y.mode, label=self._y.label)

        y = []

        # Aggregate data into groups for each annualindex
        if self._multimodel:
            for i in range(0, self._maxindex):
                y.append(trainingdate_data.data_by_index(i + 1))
        else:
            y.append(self._y.timeseries)
        # Check if each group has enough samples for the value of k_fold
        groupsize = map(len,y)
        if k_fold=='auto':
            k_fold=min(groupsize+[10])
            if k_fold==1:
                raise self.__InsufficientData(
                    "There are not enough samples for cross validation. Please provide a larger dataset"
                )
        elif k_fold==1 or not isinstance(k_fold,int):
            raise ValueError(
                "The value of k_fold must be 2 or larger."
            )
        elif not all(map(lambda x: x>=k_fold,groupsize)):
            raise self.__InsufficientData(
                "There are not enough samples for cross validation with k_fold=%s. Please choose a lower value." %k_fold
            )
        # Split each group with KFold into training and test sets
        maxsteps = k_fold+3
        t=1
        feedback_function(t,maxsteps)
        train = [pandas.Series()] * k_fold
        test = [pandas.Series()] * k_fold
        kf = KFold(n_splits=k_fold, shuffle=False)
        for i, values in enumerate(y):
            k = 0
            if len(y[i]) > 1:
                for train_index, test_index in kf.split(y[i]):
                    train[k] = train[k].append(y[i][train_index])
                    test[k] = test[k].append(y[i][test_index])
                    k += 1

        # For each KFold: train a Forecaster Object and predict the train set.
        predictions = []
        dates = []
        for i, trainingset in enumerate(train):
            t = t + 1
            feedback_function(t, maxsteps)
            fc = Forecaster(clone(self.__model[0]), FixedIndexTimeseries(trainingset, mode=self._y.mode), self._X,
                            self._laglength, self._lag, self._multimodel, self._decompose)
            fc.train()
            for target in test[i].iteritems():
                try:
                    predictions.append(fc.predict(target[0], self._X))
                    dates.append(target[0])
                except:
                    pass
        predicted_ts = FixedIndexTimeseries(pandas.Series(data=predictions, index=dates).sort_index(),mode=self._y.mode)
        targeted_ts = self._y
        t = t + 1
        feedback_function(t, maxsteps)
        self.train()
        t = t + 1
        feedback_function(t, maxsteps)
        self.Evaluator = Evaluator(targeted_ts, predicted_ts)
        return self.Evaluator

    @staticmethod
    def __no_progress(i, i_max):
        pass

    class __InsufficientData(Exception):
        pass

    class __ModelError(Exception):
        pass

class SeasonalForecaster(object):
    """Forecasting class for Seasonal Discharge, developed on the basis of the Paper
    <Statistical forecast of seasonal discharge in Central Asia using observational records: development of a generic linear modelling tool for operational water resource management.>
    by Heiko Apel et al.

    The class does gridsearch all possible feature-timeslice combinations and chooses the best n_model (default=20) models.
    Because of this approach, the training process takes much longer, up to several hours depending on the number of features to choose from.

    Attributes:
        trainingdates: a list of datetime.date objects of the periods which where used for training. Is None before training
        evaluator: An Evaluator object of the last evaluation done for this Forecaster instance. Is None before training
        trained: Boolean, False when instance has not yet been trained
    """

    def __init__(self, forecast_month, model, target, Qm, Pm=None, Tm=None, Sm=None, n_model=20, max_features=3, earliest_month=None):
        """Initialising the SeasonalForecaster Instance

            Args:
                model: A model instance that knows the method fit() and predict() for
                        a targetvector y and a feature array X
                forecast_month: Integer 1..11. Defines in which month the forecast is produced, and as such, defines what data are available.
                target: A FixedIndexTimeseries Instances which has a seasonal mode, e.g. '04-08'
                Qm (required): A FixedIndexTimeseries Instances which has a monthly mode and describes Discharge
                Pm: A FixedIndexTimeseries Instances which has a monthly mode and describes Precipitation
                Tm: A FixedIndexTimeseries Instances which has a monthly mode and describes Temperature
                Sm: A FixedIndexTimeseries Instances which has a monthly mode and describes Snowcover
                n_model (integer:1-100): The number of best models that are selected from the gridsearch.
                max_features (integer:1..8): The maximum number of features that are used for building a regression model
                        resp. complexity of the regression model. The higher this number, the longer will the grid search take.
                earliest_month (integer 1..12). Limits the first month from which on feature timewindow will be selected.
                        Default is None: The first month after the last month included in target: e.g. target mode = '04-08', earliest_month will be 9.

            Returns:
                A Forecaster instance with the methods train, predict and cross_validate

            Raises:
                ValueError: When the value for earliest_month is not within a valid range between last month of target season and forecasting_month
                ValueError: When the forecast month is not valid resp. within the allowed range of 1..first month of target season-1
                ValueError: When the values n_model exceed the valid range of 1..100
                ValueError: When the timeseries of argument target is not of mode seasonal e.g. '04-08'
                ValueError: When the timeseries of arguments Pm,Qm,Sm,Tm are not of mode monthly
            """

        if len(target.mode) < 5:
            raise ValueError("SeasonalForecast is limited to y of mode seasonal.")

        self.__model = model
        self._y = target

        self._n_model = n_model

        self._max_features = max_features

        tg_start = int(target.mode.split('-')[0])
        tg_end = int(target.mode.split('-')[1])
        target_yearswitch = False if tg_end>=tg_start else True

        if earliest_month is None:
            self._first_month = tg_end + 1 if tg_end + 1 < 13 else 1
            self._feature_year_step = False if target_yearswitch else True
        else:
            self._first_month = earliest_month

        self._last_month = forecast_month - 1 if forecast_month - 1 > 0 else 12

        if target_yearswitch:
            valid = self._first_month <= self._last_month and \
                    self._first_month > tg_end and \
                    self._last_month < tg_start
            self._feature_year_step = False
        elif tg_end < self._first_month < 13:
            if self._first_month <= self._last_month < 13:
                valid = True
                self._feature_year_step = True
            elif self._last_month < tg_start:
                valid = True
                self._feature_year_step = True
            else:
                valid = False
        elif 0 < self._first_month < tg_start:
            if self._first_month <= self._last_month < tg_start:
                valid = True
                self._feature_year_step = False
            else:
                valid = False
        else:
            valid = False

        if not valid:
            raise ValueError("The combination of earliest_month, forecast_month and target seasonal mode is not valid.")

        minyear = 0
        maxyear = 9999
        for ts in [Sm, Tm, Qm, Pm]:
            if ts:
                if ts.mode is not 'm':
                    raise ValueError("The timeseries Qm, Tm, Sm, Pm must be of monthly mode.")
                else:
                    minyear = max(ts.timeseries.index[0].year, minyear)
                    maxyear = min(ts.timeseries.index[-1].year, maxyear)

        Sm = FixedIndexTimeseries(Sm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],mode=Sm.mode, label=Sm.label) if Sm is not None else None
        Tm = FixedIndexTimeseries(Tm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],mode=Tm.mode, label=Tm.label) if Tm is not None else None
        Qm = FixedIndexTimeseries(Qm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],mode=Qm.mode, label=Qm.label) if Qm is not None else None
        Pm = FixedIndexTimeseries(Pm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],mode=Pm.mode, label=Pm.label) if Pm is not None else None

        # Create composite features
        STm = Sm.multiply(Tm, label="ST") if Sm and Tm else None
        SPm = Sm.multiply(Pm, label="SP") if Sm and Pm else None
        TPm = Tm.multiply(Pm, label="TP") if Tm and Pm else None
        STPm = STm.multiply(Pm, label="STP") if STm and Pm else None

        self._features = [Qm, Pm, Tm, Sm, STm, SPm, TPm, STPm]
        self.__feature_filter = [i for i in range(0,len(self._features)) if self._features[i]]
        self._features = [self._features[i] for i in self.__feature_filter]

        def _(message): return message
        featurenames = [
            "Y {}".format(_('disch').capitalize()),
            "Y {}".format(_('precip').capitalize()),
            "Y {}".format(_('temp').capitalize()),
            "Y {}".format(_('snow').capitalize()),
            "Y {} * {}".format(_('snow').capitalize(), _('temp').capitalize()),
            "Y {} * {}".format(_('snow').capitalize(), _('precip').capitalize()),
            "Y {} * {}".format(_('temp').capitalize(), _('precip').capitalize()),
            "Y {} * {} * {}".format(
                _('snow').capitalize(), _('precip').capitalize(), _('temp').capitalize()
            ),
        ]
        self._featurenames = [featurenames[i] for i in self.__feature_filter]

        self.__selectedmodels = None
        self._selectedfeatures = None

        self.trainingdates = None
        self.evaluator = None
        self.trained = False

    def train_and_evaluate(self, feedback_function = None):
        """Trains the model with X and y as training set

            Args:
                feedback_function (default: None): A function that is called during the execution to report on progress. Must take the arguments i (current step) and i_max(max iterations)

            Returns:
                None

            Raises:
                InsufficientData: is raised when there is not enough data to train the model for one complete year.
                    """

        if not feedback_function:
            feedback_function = self.__no_progress

        if self._feature_year_step:
            monthly_timeslices = [str(i).zfill(2) + "-" + str(i).zfill(2) for i in range(self._first_month, 13)] #OK
            monthly_timeslices += [str(i).zfill(2) + "-" + str(self._last_month).zfill(2) for i in range(self._first_month, 13)]
            monthly_timeslices += [str(i).zfill(2) + "-" + str(i).zfill(2) for i in range(1, self._last_month + 1)] #OK
            monthly_timeslices += [str(i).zfill(2) + "-" + str(self._last_month).zfill(2) for i in range(1, self._last_month)]
        else:
            monthly_timeslices = [str(i).zfill(2) + "-" + str(i).zfill(2) for i in range(self._first_month, self._last_month + 1)]
            monthly_timeslices += [str(i).zfill(2) + "-" + str(self._last_month).zfill(2) for i in range(self._first_month, self._last_month)]

        m = len(self._features)
        n = len(monthly_timeslices)


        feature_aggregates = [[feature.downsample(aggregate) for aggregate in monthly_timeslices] for feature in self._features]
        feature_aggregates_index = [[(j, i) for i in range(n)] for j in range(m)]  # tuples: (feature_index,timeslice_index)
        feature_index = range(0, m)

        qmin = len(self._features) - (self._max_features + 1)

        c = [None]*self._max_features

        for i in range(0,self._max_features):
            c[i] = combinations(feature_index, i + 1)
        feature_iterator = itertools.chain(*c)

        c = list()
        for item in feature_iterator:
            vectors = [feature_aggregates_index[i] for i in item]
            c.append(itertools.product(*vectors))
        feature_aggregate_iterator = itertools.chain(*c)

        max_iterations = int(sum([(n)**(m-q)*scisp.binom(m,m-q) for q in range(qmin+1,m+1)]) - 1)

        i=0
        error=nan
        n_model = min(self._n_model, max_iterations)
        errors = [float('inf')]*n_model
        FC_objs = [None]*n_model
        features = [None]*n_model

        #selected = [0]*max_iterations
        #errored = [0]*max_iterations
        #time = [None]*max_iterations
        #featurelength = [None] * max_iterations
        features_record = [None]*max_iterations
        for item in feature_aggregate_iterator:
            #start = timer()
            feature_list = [feature_aggregates[index[0]][index[1]] for index in item]

            if len(feature_list) > 0:
                try:
                    FC_obj = Forecaster(self.__model, self._y, feature_list, lag=0, laglength=[1] * len(feature_list),multimodel=False, decompose=False)
                    CV = FC_obj.train_and_evaluate()
                    error = mean(CV.computeRelError())
                    if error < max(errors):
                        #selected[i] = 1
                        index = errors.index(max(errors))
                        errors[index] = error
                        FC_objs[index] = FC_obj
                        features[index] = [None] * len(self._features)
                        for k in item:
                            features[index][k[0]] = monthly_timeslices[k[1]]
                except:
                    #errored[i] = 1
                    error = nan

                # ...
                #time[i] = timer() - start
                #features_record[i] = item
                #featurelength[i] = len(feature_list)

                i = i + 1

                feedback_function(i,max_iterations)

        if FC_objs.count(None) > 0:
            raise self.__InsufficientData("There is not enugh data to return the number of requested models.")
        else:
            self.__selectedmodels = FC_objs
            self._selectedfeatures = features
            self.evaluator = SeasonalEvaluator(self._featurenames, features, [model.Evaluator for model in self.__selectedmodels])
            self.trainingdates = list(set().union(*[model.trainingdates for model in FC_objs]))
            self.trained = True
            return self.evaluator

    def predict(self, targetdate, Qm, Pm=None, Tm=None, Sm=None):
        """Does a prediction with the trained model

            Args:
                targetdate: A datetime.date that is within the season to be forecasted.
                Qm (required): A FixedIndexTimeseries Instances which has a monthly mode and describes Discharge. Needs to contain the necessary data to predict targetdate
                Pm: A FixedIndexTimeseries Instances which has a monthly mode and describes Precipitation. Needs to contain the necessary data to predict targetdate
                Tm: A FixedIndexTimeseries Instances which has a monthly mode and describes Temperature. Needs to contain the necessary data to predict targetdate
                Sm: A FixedIndexTimeseries Instances which has a monthly mode and describes Snowcover. Needs to contain the necessary data to predict targetdate

            Returns:
                A list of length=n_model with all predictions as float. If the prediction of one model failed e.g. due to missing feature data, a value of nan is inserted

            Raises:
                ValueError: is raised when the Qm,Pm, Tm or Sm are not of monthly mode 'm'
                InsufficientData: If not all the timeseries of Qm,Pm, Tm or Sm, that were given as argument in init call are passed to this function.
                    """
        if not self.trained:
            raise self.__ModelError("The model has not been trained yet.")

        for item in [Qm, Pm, Tm, Sm]:
            if item is not None and item.mode != 'm':
                raise ValueError("features must be of mode 'm")

        STm = Sm.multiply(Tm) if Sm and Tm else None
        SPm = Sm.multiply(Pm) if Sm and Pm else None
        TPm = Tm.multiply(Pm) if Tm and Pm else None
        STPm = STm.multiply(Pm) if STm and Pm else None

        features = [Qm, Pm, Tm, Sm, STm, SPm, TPm, STPm]
        features = [features[i] for i in self.__feature_filter]
        if features.count(None) > 0:
            raise self.__InsufficientData("One of the required feature datasets is missing")

        pred=list()
        for i,FC_obj in enumerate(self.__selectedmodels):
            featureindex = self._selectedfeatures[i]
            feature_list = [features[i].downsample(x) if x is not None else None for i, x in enumerate(featureindex)]
            feature_list = filter(None, feature_list)
            try:
                pred.append(FC_obj.predict(targetdate,feature_list))
            except:
                pred.append(nan)
        return(pred)

    def update(self, target, Qm, Pm=None, Tm=None, Sm=None):

        if not target.mode == self._y.mode:
            raise ValueError("The updated target data timeseries is of different mode than the original dataset")

        minyear = 0
        maxyear = 9999
        for ts in [Sm, Tm, Qm, Pm]:
            if ts:
                if ts.mode is not 'm':
                    raise ValueError("The timeseries Qm, Tm, Sm, Pm must be of monthly mode.")
                else:
                    minyear = max(ts.timeseries.index[0].year, minyear)
                    maxyear = min(ts.timeseries.index[-1].year, maxyear)

        Sm = FixedIndexTimeseries(Sm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],
                                  mode=Sm.mode, label=Sm.label) if Sm is not None else None
        Tm = FixedIndexTimeseries(Tm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],
                                  mode=Tm.mode, label=Tm.label) if Tm is not None else None
        Qm = FixedIndexTimeseries(Qm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],
                                  mode=Qm.mode, label=Qm.label) if Qm is not None else None
        Pm = FixedIndexTimeseries(Pm.timeseries[datetime.date(minyear, 1, 1):datetime.date(maxyear, 12, 31)],
                                  mode=Pm.mode, label=Pm.label) if Pm is not None else None
        # Create composite features
        STm = Sm.multiply(Tm, label="ST") if Sm and Tm else None
        SPm = Sm.multiply(Pm, label="SP") if Sm and Pm else None
        TPm = Tm.multiply(Pm, label="TP") if Tm and Pm else None
        STPm = STm.multiply(Pm, label="STP") if STm and Pm else None

        features = [Qm, Pm, Tm, Sm, STm, SPm, TPm, STPm]
        features = [features[i] for i in self.__feature_filter]
        if features.count(None) > 0:
            raise self.__InsufficientData("One of the required feature datasets is missing")
        else:
            self._y = target
            self._features = features

        return self.retrain()

    def retrain(self):

        if not self.trained:
            raise self.__ModelError("The model has not been trained yet.")

        FC_objs = [None]*len(self.__selectedmodels)
        for i in range(len(self.__selectedmodels)):
            featureindex = self._selectedfeatures[i]
            feature_list = [self._features[k].downsample(x) if x is not None else None for k, x in enumerate(featureindex)]
            feature_list = filter(None, feature_list)
            try:
                FC_objs[i] = Forecaster(self.__model, self._y, feature_list, lag=0, laglength=[1] * len(feature_list),
                                    multimodel=False, decompose=False)
                FC_objs[i].train_and_evaluate()
            except:
                raise self.__ModelError("There was an Error training the model.")

        self.__selectedmodels = FC_objs
        self.evaluator = SeasonalEvaluator(self._featurenames, self._selectedfeatures,
                                           [model.Evaluator for model in self.__selectedmodels])
        self.trainingdates = list(set().union(*[model.trainingdates for model in FC_objs]))

        return self.evaluator

    class __ModelError(Exception):
        pass

    class __InsufficientData(Exception):
        pass

    @staticmethod
    def __no_progress(i, i_max):
        pass

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


