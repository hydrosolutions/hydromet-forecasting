
# hydromet forecasting

## Introduction

The library is designed to automize and enhance the forecasting method, that the hydrometeorological agency in Kyrgyzstan applies for pentadal (5-day), decadal (10-day), monthly and seasonal timeseries of their river basins. Originally, these forecasts are produced manually, using MS Excel and expert knowledge. The module has been developed with the goal, to digitize the manual procedure and give space for some more experiments with additional data sources like snow timeseries and machine learning methods.
The library has two distinguished methods for normal, continuous forecasts like monthly and another method (grid-search) for seasonal forecasting. The latter was implemented on the basis of this research paper by Heiko Apel et. al: https://www.hydrol-earth-syst-sci.net/22/2225/2018/

## Overview
The library is split into three modules: timeseries.py, forecasting.py and evaluating.py

Setting up a forecast & predicting:

1. Loading all relevant data as FixedIndexTimeseries instances from the timeseries.py module
2. Initialising a RegressionModel instance  from the forecasting.py module with default or custom parameters.
3. Initialising a Forecaster or Seasonal Forecaster instance from the forecasting.py module. Here, most of the relevant parameters are set. The RegressionModel from the previous step is given as argument for initialisation.
4. Now, with the newly created Forecaster instance, the method train_and_evaluate is executed. This is the computationally most expensive step. The method will return an Evaluator resp. Seasonal Evaluator instance from the evaluating.py module. The Evaluator instance is as well stored as a Forecaster instance attribute .evaluator
5. The Evaluator knows the method write_html(filename). It prints a perfromance assessment as html to the given filepath.
6. A prediction is computed with the method predict(targetdate, X) of the trained Forecaster instance. X is the feature data.

### timeseries.py
 Unfortunately, the pandas library does not support timeseries with decadal or pentadal frequency. For this reason, a new class was designed, that wraps around pandas to handle such frequencies.
 
#### FixedIndexTimeseries
FixedIndex means, that each year has the same number of periods and that every period takes the same position in every year, e.g. monthes or semi-monthes etc. It does not work for timeseries with periods, that strictly consist of the same number of days like "weeks" and as such, might overlap New Year.
In this class, the attribute "timeseries" contains the pandas dataframe. Here, every datapoint is assigned a date, which is the first day of the timeperiod that it describes. E.g. in monthly timeseries datetime.date(2011,6,1) is the timestamp of the datapoint for June 2011.

A timeseries can be read by using the child class FixedIndexTimeseriesCSV. When initialising, a mode must be given and can be either:

* 'p' for pentadal data
* 'd' for decadal data
* 'm' for monthly data 
* 'dl' for daily data (day 366 is ignored for simplicity)
* 'xx-yy' for seasonal data, whereby xx is the first month and yy is the second month as two digit integer: e.g. '04-09' for April to and including September

The CSV file must be formatted in the following way:

* Rows contain the data of 1 year.
* The first cell in each row contains the year e.g. 2018 of that row. 
* The length of the rows corresponds to the number of periods, e.g. monthly 12+1(yearvalue)
* Empty cells and strings are read as NaN


~~~~ 
2010,21.6,21.4,23.1,31.8,20.6,45.3,25.2,11.3,23.9,29.6,28.1,27
2011,23.3,22.7,24.9,26.6,18,31.7,15.1,,20.8,26.8,28.9,26.7
2012,23.8,,22.1,15.6,2.7,10.4,6.4,3.3,11.8,12.4,19.4,21.6
~~~~

The class offers several method to manipulate the timeseries, extract information or handle the frequency mode. 

Load a csv:
```python
precipitation=FixedIndexTimeseriesCSV("example_data/monthly/P.csv",mode="m")
```

The timeseries frequency can be downsampled to a lower frequency, e.g. from monthly to seasonal (April-September) by calling:

```python
seasonal_precipitation = precipitation.downsample(mode='04-09')
```

### forecasting.py

#### Regression Model (sklearn Estimator)

This class helps to initialise a sklearn estimator from a selection. At the moment, only linear regression, decision tree regression and the lasso regression estimators are enabled. It is possible to add other estimator classes, but for the use case here the mentioned estimators are sufficient.

An estimator (here called model) is initialised in the following way:

```python
# List available nmodels resp. estimators
print(RegressionModel.SupportedModels.list_models())

# Initialise a decision tree model class
reg_model = RegressionModel.build_regression_model(RegressionModel.SupportedModels(3))

# Print default model parameters:
print("Default parameters: %s" %reg_model.default_parameters)

# Print possible parameter choices:
print("Possible parameters or range: %s" %reg_model.selectable_parameters)

# Set parameter and configure the regression model from the model class
model=reg_model.configure({'n_estimators':20})  
```


#### Forecaster
This is the class for general forecasts resp. everything with lower frequency than seasonal timeseries. It is initialised in minimum with an Regression Model Instance (see above), a target timeseries y (FixedIndexTimeseries Instance, see above), a list of feature timeseries X and a list of laglengths that correspond to the features. The laglength defines how many timelags of a feature are included in the forecast, e.g. for a monthly timeseries a value of 3 means the last 3 month. Additional parameters let you specify more details.

*(Remark: y is the target, which is the value that will be predicted. X are the features, the data that are used to predict y. With timeseries, the values from the past are used to predict values from the future. This is the reason, why the dataset discharge can be both, target and feature. Previous values of discharge (how many is defined by the argument laglength) are used to predict its next value, together with past values of temperature and precipitation.)*

A basic initialisation is:
```python
discharge=FixedIndexTimeseriesCSV("example_data/decadal/Ala_Archa_short/Q.csv",mode="d")
precipitation=FixedIndexTimeseriesCSV("example_data/decadal/Ala_Archa_short/P.csv",mode="d")
temperature=FixedIndexTimeseriesCSV("example_data/decadal/Ala_Archa_short/T.csv",mode="d")
FC_obj = Forecaster(model=model,y=discharge,X=[discharge,temperature,precipitation],laglength=[3,3,3])
```


In order to train this model, the method train_and_evaluate() is called. It returns an Evaluator (see below) instance, with which one can write an html report of the model performance. The model performance is assessed by a k-fold cross validation on is done using all available data. Depending on the amount of available data and complexity of the model, this process might take a while. The train_and_evaluate() function takes the argument feedback_function, which is triggered every step, e.g. can report on the current state of the computation. A valid feedback_function must take the argument i and i_max, whereby i is the current step and i_max is the maximal, final step.
```python
def print_percentage(i, i_max):
    print(int(100*i/i_max)
    
PA_obj = FC_obj.train_and_evaluate(feedback_function=print_percentage)
PA_obj.write_html("assessment_report.html")
```

Finally, in order to make a prediction with the trained model. the function predict() is called.

```python
# forecast y for the first decade of June 2011
prediction = FC_obj.predict(targetdate=datetime.date(2011,6,1),X=[discharge,temperature,precipitation])
```
 Two arguments need to be given:
 
* A targetdate as datetime.date object: The targetdate defines the targetperiod. It points to a date which is within that targetperiod, but it does not matter if it is its first or last or any other date within the period. E.g. datetime.date(2011,6,1), datetime.date(2011,6,12), datetime.date(2011,6,30) all point to the timeperiod of June for a target timeseries of monthly frequeny resp. mode='m'.
* X, a list of featuredata in the same format as it was when initialising the Forecaster instance. The FixedIndexTimeseries given here usually contain newer data as when initialising the Forecaster instance. They do not need to contain all data available, but might only contain the data that is required for that specific forecasting setup (laglength, etc.) and targetdate. All additional data is ignored. If not all required data is given, the function will raise and InsufficientData Exception.

#### SeasonalForecaster

This Forecaster class has been designed to enhance the results of seasonal forecasts, where it is much more difficult to reach sufficient performance. A grid search tests all feature-timewindow combinations and stores the best 20 models. For more details, read:
>"Statistical forecast of seasonal discharge in Central Asia using observational records: development of a generic linear modelling tool for operational water resource management " by H. Apel et. al (https://www.hydrol-earth-syst-sci.net/22/2225/2018/)

A SeasonalForecaster instance is initialised by: 
```python
Talas_Q=FixedIndexTimeseriesCSV("example_data/monthly/Talas_Kluchevka/Q.csv",mode="m") #Discharge
Talas_P=FixedIndexTimeseriesCSV("example_data/monthly/Talas_Kluchevka/PREC_ERA.csv",mode="m") #Precipitation
Talas_T=FixedIndexTimeseriesCSV("example_data/monthly/Talas_Kluchevka/TEMP_ERA.csv",mode="m") #Temperature
Talas_S = FixedIndexTimeseriesCSV("example_data/daily/Talas_Kluchevka/SNOW.csv",mode="dl") #Snow Cover (daily)
Talas_S = Talas_S.downsample(mode='m')
target = Talas_Q.downsample('04-09')

FC_obj = SeasonalForecaster(model=model, target=target, Qm=Talas_Q, Pm=Talas_P, Sm=Talas_S, Tm=Talas_T, forecast_month=4)

```

* model: A Regression Model Instance (see above)
* target: a FixedIndexTimeseries of seasonal mode, e.g. mode='04-09' for April to and including September
* Qm, Pm, Sm, Tm are the feature timeseries and must have mode='m' resp. be of monthly frequency. Only Qm is stricly required, the others can be None, although the model performance will be very low in that case.
* forecast_month: The month when the forecast is produced as integer. Defines which data are available to the model. If forecast_month=3, data from February and earlier are available to the model. Must be smaller or equal to the first month of the target season.
* TODO explain other, optional arguments


Training, evaluating and predicting is similar to the general Forecaster class. Depending on the model complexity,  train_and_evaluate() might take a few hours:
```python
PA_obj = FC_obj.train_and_evaluate()
PA_obj.write_html("assessment_report.html")

prediction=FC_obj.predict(targetdate=datetime.date(2014,4,1),Qm=Talas_Q,Pm=Talas_P,Sm=Talas_S,Tm=Talas_T)
print(prediction)
```


### evaluating.py

#### Evaluator & SeasonalEvaluator

An instance of those classes is returned when calling the method train_and_evaluate() of a Forecaster resp. SeasonalForecaster Instance. 
The most important method is write_html(), which writes an assessment report to a specified path.

```python
PA_obj = FC_obj.train_and_evaluate()
PA_obj.write_html(filename="assessment_report.html")
```

At the moment, the format of the report is defined by a static template.
