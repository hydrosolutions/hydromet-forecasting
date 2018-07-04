import datetime

import enum
import pandas
from numpy import nan, array, isnan, full, nanmean, mean
from sklearn.base import clone
from sklearn.model_selection import KFold

from hydromet_forecasting.timeseries import FixedIndexTimeseries
from hydromet_forecasting.evaluating import Evaluator

from sklearn import preprocessing
from monthdelta import monthdelta

from stldecompose import decompose as decomp
import itertools

import scipy.special as scisp

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
        trainingdates: a list of datetime.date objects of the periods whcih where used for training. Is None before training
        evaluator: An Evaluator object of the current training state of the Forecaster instance. Is None before training
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

        self._model = clone(model)
        self._multimodel = multimodel

        if not self._multimodel:
            self._maxindex = 1
            self._model = [self._model]
        else:
            self._maxindex = y.maxindex
            self._model = [clone(self._model) for i in range(self._maxindex)]


        self._decompose = decompose
        self._seasonal = [0 for i in range(y.maxindex)]

        self._y = y
        self._y.timeseries.columns = ["target"]

        if type(X) is not list:
            self._X = [X]
        else:
            self._X = X

        self._X_type = [x.mode for x in self._X]

        self._lag = -lag # switches the sign of lag as argument, makes it easier to understand

        if type(laglength) is not list:
            self._laglength = [laglength]
        else:
            self._laglength = laglength

        if not len(self._laglength) == len(X):
            raise ValueError("The arguments laglength and X must be lists of the same length")


        self._y_scaler = [preprocessing.StandardScaler() for i in range(self._maxindex)]
        self._X_scaler = [preprocessing.StandardScaler() for i in range(self._maxindex)]

        assert len(self._X) > 0, "predictor dataset must contain at least one feature"

        self.trainingdates = None
        self.evaluator = None

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
            x_set = self._X_scaler[i].fit_transform(array(X_list[i]))
            y_set = self._y_scaler[i].fit_transform(array(y_list[i]).reshape(-1,1))

            if len(y_set) > 0:
                try:
                    self._model[i].fit(x_set, y_set.ravel())
                except Exception as err:
                    print(
                        "An error occured while training the model for annual index %s. Please check the training data." % (
                            i + 1))
                    raise err
            else:
                raise self.InsufficientData(
                    "There is not enough data to train the model for the period with annualindex %s" % (i + 1))

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
        if not type == self._X_type:
            raise ValueError(
                "The input dataset X must be a list of FixedIndexTimeseries objects with type and length %s" % self._X_type)

        featuredates = self._aggregate_featuredates(targetdate)
        X_values = self._aggregate_features(featuredates, X)

        if self._multimodel:
            annual_index = self._y.convert_to_annual_index(targetdate)
        else:
            annual_index = 1

        if not self.trainingdates:
            raise self.ModelError(
                "There is no trained model to be used for a prediciton. Call class method .train() first.")
        elif isnan(X_values).any():
            raise self.InsufficientData("The data in X is insufficient to predict y for %s" % targetdate)
        else:
            x_set = self._X_scaler[annual_index - 1].transform(X_values.reshape(1, -1))
            prediction = self._model[annual_index - 1].predict(x_set)
            invtrans_prediction = self._y_scaler[annual_index - 1].inverse_transform(prediction.reshape(-1,1))
            return float(invtrans_prediction+self._seasonal[self._y.convert_to_annual_index(targetdate)-1])

    def cross_validate(self, k_fold='auto'):
        """Conducts a crossvalidation on the Forecaster instance and returns an Evaluator instance.

            Is used to measure the performance of the forecast. Uses the scikit K-Folds cross-validator without
            shuffling.

            Args:
                k_fold: 'auto'(default) or an integer > 1, Defines in how many train/test sets the data are split. A small value
                        is better to measure real model performance but takes much longer to compute. 'auto' will choose 10 splits
                        or smaller depending on the size of the available dataset.
            Returns:
                An instance of the Evaluator class.

            Raises:
                ValueError: if k_fold is set to a value of <1 or is not an integer.
                InsufficientData: is raised when the dataset contains too few samples for crossvalidation with
                                    the chosen value of k_fold or if it only contains one sample.
            """
        #self.indicate_progress(0)
        y = []

        # Aggregate data into groups for each annualindex
        if self._multimodel:
            for i in range(0, self._maxindex):
                y.append(self._y.data_by_index(i + 1))
        else:
            y.append(self._y.timeseries)
        # Check if each group has enough samples for the value of k_fold
        groupsize = map(len,y)
        if k_fold=='auto':
            k_fold=min(groupsize,10)
            if k_fold==1:
                raise self.InsufficientData(
                    "There are not enough samples for cross validation. Please provide a larger dataset"
                )
        elif k_fold==1 or not isinstance(k_fold,int):
            raise ValueError(
                "The value of k_fold must be 2 or larger."
            )
        elif not all(map(lambda x: x>=k_fold,groupsize)):
            raise self.InsufficientData(
                "There are not enough samples for cross validation with k_fold=%s. Please choose a lower value." %k_fold
            )
        # Split each group with KFold into training and test sets
        #maxsteps = len(y)+k_fold*10+1
        #t=1
        #self.indicate_progress(float(t)/maxsteps*100)
        #t+1
        train = [pandas.Series()] * k_fold
        test = [pandas.Series()] * k_fold
        kf = KFold(n_splits=k_fold, shuffle=False)
        for i, values in enumerate(y):
            #self.indicate_progress(float(t) / maxsteps * 100)
            #t=t + 1
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
            #self.indicate_progress(float(t) / maxsteps * 100)
            #t = t + 10
            fc = Forecaster(clone(self._model[0]), FixedIndexTimeseries(trainingset, mode=self._y.mode), self._X,
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
        return Evaluator(targeted_ts, predicted_ts)

    def indicate_progress(self,p):
        print("progress is %s%%" %(p))

    class InsufficientData(Exception):
        pass

    class ModelError(Exception):
        pass


class SeasonalForecast(object):
    """Forecasting class for Seasonal Discharge, developed on the basis of the Paper
    <Statistical forecast of seasonal discharge in Central Asia using observational records: development of a generic linear modelling tool for operational water resource management.>
    by Heiko Apel et al.

    The class does gridsearch all possible feature-timeslice combinations and chooses the best n_model (default=20) models.
    Because of this approach, the training process takes much longer, up to several hours depending on the number of features to choose from.

    Attributes TODO :
        model: a list of datetime.date objects of the periods whcih where used for training. Is None before training
        evaluator: An Evaluator object of the current training state of the Forecaster instance. Is None before training
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
            """
        if len(target.mode) < 5:
            raise ValueError("SeasonalForecast is limited to y of mode seasonal.")

        self.model = model
        self.y = target

        if 0 < n_model < 101:
            self.n_model = n_model
        else:
            raise ValueError("n_model must be in the range 1...100")

        self.max_features = max_features

        if  target.mode.split('-')[0] > forecast_month > 0:
            self.last_month = forecast_month-1
        else:
            raise ValueError("The argument forecast_month does need to be in between 0 and the first month of the forecasted season")

        self.feature_year_step = True
        if earliest_month is None:
            self.first_month = int(target.mode.split('-')[1])+1
        elif int(target.mode.split('-')[1]) < earliest_month < 12:
            self.first_month = earliest_month
        elif self.last_month >= earliest_month > 0:
            self.feature_year_step = False
            self.first_month = earliest_month
        else:
            raise ValueError('The argument earliest_month is not valid.')

        # Create composite features
        STm = Sm.multiply(Tm) if Sm and Tm else None
        SPm = Sm.multiply(Pm) if Sm and Pm else None
        TPm = Tm.multiply(Pm) if Tm and Pm else None
        STPm = STm.multiply(Pm) if STm and Pm else None

        self.features = [Qm,Pm,Tm,Sm,STm,SPm,TPm,STPm]
        self._featurenames = ["Qm", "Pm", "Tm", "Sm", "STm", "SPm", "TPm", "STPm"]

        self._selectedmodels = None
        self._selectedfeatures = None
        self._score = None

    def train(self, feedback_function = None):
        """Trains the model with X and y as training set

            Args:
                feedback_function (default: None): A function that is called during the execution to report on progress. Must take the arguments i (current step) and i_max(max iterations)

            Returns:
                None

            Raises:
                InsufficientData: is raised when there is not enough data to train the model for one complete year.
                    """

        if not feedback_function:
            feedback_function = self.no_progress

        if self.feature_year_step:
            monthly_timeslices = [str(i).zfill(2) + "-" + str(i).zfill(2) for i in range(self.first_month, 13)]
            monthly_timeslices += [str(self.first_month).zfill(2) + "-" + str(i).zfill(2) for i in range(self.first_month+1, 13)]
            monthly_timeslices += [str(i).zfill(2) + "-" + str(i).zfill(2) for i in range(1, self.last_month)]
            monthly_timeslices += [str(self.first_month).zfill(2) + "-" + str(i).zfill(2) for i in range(1, self.last_month)]
        else:
            monthly_timeslices = [str(i).zfill(2) + "-" + str(i).zfill(2) for i in range(self.first_month, self.last_month + 1)]
            monthly_timeslices += [str(self.first_month).zfill(2) + "-" + str(i).zfill(2) for i in range(self.first_month + 1, self.last_month + 1)]

        n = len(monthly_timeslices)
        feature_aggregates = []
        feature_aggregates_index = []
        for feature in filter(None,self.features):
            feature_aggregates.append([self.downsample_helper(feature,aggregate) for aggregate in monthly_timeslices])
            feature_aggregates_index.append([None]+range(0,len(monthly_timeslices)))


        k = len(feature_aggregates)
        qmin = len(filter(None,self.features)) - (self.max_features+1)
        feature_iterator = itertools.product(*feature_aggregates_index)
        feature_iterator = itertools.ifilter(lambda x: qmin < x.count(None), feature_iterator)
        # formula for the number of possible combinations, tough brain work to find out *sweating..., -1 to substract (None,None,None,...)
        max_iterations = sum([(n)**(k-q)*scisp.binom(k,k-q) for q in range(qmin+1,k+1)]) - 1

        i=0
        score=nan
        scores = [float('inf')]*self.n_model
        FC_objs = [None]*self.n_model
        features = [None]*self.n_model

        for item in feature_iterator:

            feature_list = map(lambda x: feature_aggregates[x][item[x]] if item[x] is not None else None, range(0,len(feature_aggregates)))
            feature_list = filter(None,feature_list)

            if len(feature_list) > 0:
                FC_obj = Forecaster(self.model, self.y, feature_list,lag=0, laglength=[1]*len(feature_list), multimodel=False, decompose=False)
                try:
                    CV = FC_obj.cross_validate()
                    score = mean(CV.computeRelError())
                    if score < max(scores):
                        index = scores.index(max(scores))
                        scores[index] = score
                        FC_objs[index] = FC_obj
                        features[index] = [monthly_timeslices[k] if k is not None else None for k in item]
                except:
                    score = nan

                i = i + 1
                feedback_function(i,max_iterations)


        for model in FC_objs:
            model.train()

        self._selectedmodels = FC_objs
        self._selectedfeatures = features
        self._score = scores
        return None

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

        for item in [Qm, Pm, Tm, Sm]:
            if item is not None and item.mode is not 'm':
                raise ValueError("features must be of mode 'm")

        STm = Sm.multiply(Tm) if Sm and Tm else None
        SPm = Sm.multiply(Pm) if Sm and Pm else None
        TPm = Tm.multiply(Pm) if Tm and Pm else None
        STPm = STm.multiply(Pm) if STm and Pm else None

        features = [Qm, Pm, Tm, Sm, STm, SPm, TPm, STPm]
        for i, feature in enumerate(features):
            if self.features[i] is not None and feature is None:
                raise self.InsufficientData("The feature "+self._featurenames[i]+" was not found in arguments")
        features = filter(None, features)
        pred=list()
        for i,FC_obj in enumerate(self._selectedmodels):
            featureindex = self._selectedfeatures[i]
            feature_list = [self.downsample_helper(features[i],x) if x is not None else None for i,x in enumerate(featureindex)]
            feature_list = filter(None, feature_list)
            try:
                pred.append(FC_obj.predict(datetime.date(2011,4,1),feature_list))
            except:
                pred.append(nan)
        return(pred)

    @staticmethod
    def downsample_helper(timeseries,mode):
        """ A workaround for FixedIndexTimeseries of mode seasonal that overlap new year e.g. '11-02', which is not natively handled by that class.

            The returned FixedIndexTimeseries has mode '01-x' instead of e.g. '11-x', but the aggegrated data are averaged over the full timewindow.
                    """
        res = mode.split("-")
        if int(res[0]) > int(res[1]):
            mode1 = res[0]+'-12'
            weight1 = 13-int(res[0])
            aggregate1 = timeseries.downsample(mode1)
            shifted_index = aggregate1.timeseries.index.values + monthdelta(weight1)
            aggregate1.timeseries.index = shifted_index

            mode2 = '01-'+res[1]
            weight2 = int(res[1])
            aggregate2 = timeseries.downsample(mode2)

            timeseries = (aggregate1.timeseries*weight1).add(aggregate2.timeseries*weight2)/(weight1+weight2)
            return FixedIndexTimeseries(timeseries, mode=mode2, label=mode)
        else:
            return timeseries.downsample(mode)

    class ModelError(Exception):
        pass

    class InsufficientData(Exception):
        pass

    @staticmethod
    def no_progress(i, i_max):
        pass




