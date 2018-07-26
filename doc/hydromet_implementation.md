
# hydromet-forecast guideline for user interaction

## 1. General Forecast (Monthly, Decadal, 5-day)

### Parameter choice by user
The following parameters should be free to choose by the user within the specified range. name

#### model:
see section [Regression Model](#regression-model)

#### X
**label**: Feature data
**description**: A list of all variables that are used to predict the data in y
**type**: list of FixedIndexTimeseries instances
**user choice**: A selection of discharge, precipitation, temperature or snow data from all available stations.
**default**: the data in y (the variable that will be forecasted -> autoregression)

#### laglength
**label**: feature timelag
**description**: describes how much past data of each variable in X is used to predict y. Absolute timespan depends on the frequency of X and laglength. E.g. a laglength of 3 for monthly data equals 3 month into the past.
**type**: list of integers of same length as X
**user choice**: for each selected data in X, an integer from 1 to 10
**default**: [1,1,...]

#### lag
**label**: Issue Date
**description**: this parameters is indirectly defined by the user choice of the issue date of the forecast. The parameter defines, which timespan of data can be used features. 
E.g. lag=0: The forecast is issued on the first day of the predicted period. lag=-1: The forecats is issud one day before the forecasted period begin and so on. This has the consequence, that the data from the period directly preceeding the forecasted period can not be used as feature, because it is not complete resp. the last day of data is missing. The selection of an issue date does not show on the forecast page, but on the inital organisation setup page.
**default**: No default, an issue date must be set during the initial organisation setup.

#### multimodel
**label**: multimodel
**description**: Seperate regression model is trained individually for each period of the year (TRUE) vs. only one regression model is trained (FALSE). Might improve the forecast if timeseries has strong seasonal characteristics.
**type**: boolean
**user choice**: ON/OFF
**default**: OFF

### fixed Parameters

#### decompose
value=FALSE


## Seasonal Forecasts
forecast_month, model, target, Qm, Pm=None, Tm=None, Sm=None, n_model=20, max_features=3, earliest_month=None

### Parameter choice by user
#### model:
see section [Regression Model](#regression-model)

#### forecast_month
**label**: Issue Date
**description**: The issue date of the seasonal forecast. Can be a specific day like 5th of April. But the parameters will just take a value of 4. The day within the month thus does not matter to the method
**type**: integer 1..12
**user choice**: January to August, e.g. 1 to 8
**default**: No default, an issue date must be set during the initial organisation setup.

#### Pm,Tm,Sm
**label**: Feature Data
**description**: Seasonal forecasts are different to general forecasts. It requires at least the discharge data of the hydro station that will be forecasted (parameter Qm) and can additionally include at most one additional timeseries for temperature (Tm), precipitation (Pm) and Snow (Sm). The timeseries are FixedIndexTimeseries Instances strictly of mode "monthly". Use class method downsample(mode='m') if required.
**type**: for each parameters, one FixedIndexTimeseries Instances strictly of mode "monthly"
**user choice**: for each parameter, select the hydro or meteostation that shall be used.
**default**: None

#### n_model
**label**: Ensemble size
**description**: Seasonal forecasts use ensembles resp. multiple models to predict the value. This parameter defines how many models will be selected.
**type**: integer
**user choice**: 1..100
**default**: 20

#### max_features
**label**: Feature limit per model
**description**: Limits how many features can be included in one model. This parameters strongly influences how long the training process takes, because a larger value will allow more models to select from.
**type**: integer
**user choice**: 1..4
**default**: 2

### fixed Parameters
#### earliest_month
**value**:11


## Regression Model

The user selects a regression Model type from the list of supported models. For the selected model, the user can now select the parameters specific to this model or use the default values

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


