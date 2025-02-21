# Time series forecasting with FEDOT. Guide
#### FEDOT version = 0.7.2


```python
!pip install fedot==0.7.2
```


--- 

"Nothing is clear but very interesting." - our lab's motto (but hope this tutorial will help you figure it out)

## Introduction 

A time series is a sequence of values ordered by time. For example, we can fix the value of a parameter at points in time $t, t+1, t+2$, etc. Various data can be represented as time series, e.g. the air temperature, the number of sales in the online store, the number of people who watch the stream on youtube, and much more. This series will look something like this (fig. 1).


Figure 1. What is time series and autoregression

An important feature of the time series is the "autoregression". As can be seen from fig. 1, elements from the past can have connections with future elements. The closer these elements are to each other, the stronger the connection, this is not always the case but very often. So, autoregression is the relationship between the elements of the time series. This feature of time series is used to predict future values when we know only past and current values. This task is called a time series forecasting.

Time series forecasting can be solved using autoregressive models from Python libraries, such as [statsmodels](https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressions.html). However, this is still a challenging task. First of all, because even though time series forecasting models can give high results, but often require a researcher with high qualifications to adjust the hyperparameters.

But things get easier if you use FEDOT. Let's look at how values in time series can be predicted.

## Imports


```python
# Additional imports are required
import os 

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Plots
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

import warnings
warnings.filterwarnings('ignore')

# Prerocessing for FEDOT
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

# FEDOT 
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode

import logging
logging.raiseExceptions = False
```

Let's take a closer look at what we imported from the FEDOT framework. The framework's functionality for time series forecasting is based on autoregressive models. We will use them below. The main meaning of such models can be shown in this picture (fig. 2)

<img src="../jupyter_media/time_series/autoregressive_models.png" width="1000"/> 

Figure 2. Autoregressive models for time series forecasting

let's look at how such an autoregressive function works in a simple example


```python
def simple_autoregression_function(dataframe):
    """
    The function trains a linear autoregressive model, 
    which use 2 predictors to give a forecast
    
    :param dataframe: pandas DataFrame with time series in "Values" column

    :return model: fitted linear regression model
    """
    
    # Take only values series
    vals = dataframe['Values']
    
    # Convert to lagged form (as in the picture above)
    lagged_dataframe = pd.concat([vals.shift(2), vals.shift(1), vals], axis=1)
    
    # Rename columns
    lagged_dataframe.columns = ['Predictor 1', 'Predictor 2', 'Target']
    
    # Drop na rows (first two rows)
    lagged_dataframe.dropna(inplace=True)
    
    # Display first five rows
    print(lagged_dataframe.head(4))
    
    # Get linear model to train
    model = LinearRegression()
    
    # Prepare features and target for model
    x_train = lagged_dataframe[['Predictor 1', 'Predictor 2']]
    y_train = lagged_dataframe['Target']
    
    # Fit the model
    model.fit(x_train, y_train)
    return model
```

let's build a simple model on a short time series.


```python
# Prepare simple time series
example_dataframe = pd.DataFrame({'Values':[5,7,9,7,5,5,3,4,6,14,6,3,5]})

# Get fitted model
fitted_model = simple_autoregression_function(example_dataframe)

# Now we can forecast the values by knowing 
# the current and previous values (2 predictors in total)
predictors = [3,5]
print(f'\nPredictors: {predictors}')
forecast = fitted_model.predict([predictors])

print(f'Forecasted value: {forecast}')
```

       Predictor 1  Predictor 2  Target
    2          5.0          7.0       9
    3          7.0          9.0       7
    4          9.0          7.0       5
    5          7.0          5.0       5
    
    Predictors: [3, 5]
    Forecasted value: [7.51349541]
    

As can be seen from the example, in order to model the dependence between the elements of a time series, autoregressive models are needed. Here we have used linear regression, but we can approximate the dependence using nonlinear models (random forest for example) also.

But these models are not used directly in the framework. They are placed in containers called "nodes".

## Node
A node is where the model is placed.

In one such node, as in a container, we can put one model, which we want (the list of models will be presented below).

##### You can do it using a simple action:
    node_first = PrimaryNode('linear')
    
There are two types of nodes: 
* PrimaryNode - node where predictors are the source data;
* SecondaryNode - the node in which the predictors, prediction of the other nodes.

Schematically, it looks like this (fig. 3).

<img src="../jupyter_media/time_series/data_flow.png" width="900"/>

Figure 3. Differences between PrimaryNode and SecondaryNode. Demonstration of data flow in the pipeline.

## Pipeline
But using only single models is too easy. FEDOT can do much more. We can combine the nodes into pipelines - let's use Pipeline class.

Thus, if the model is wrapped in a node, the nodes are wrapped in pipeline. Below we will look in detail at how you can build such pipelines and what forecasts they can make.

<img src="../jupyter_media/time_series/chains_example.png" width="900"/>

Figure 4. Two examples of constructed pipelines

## Synthetic example

Below we will look at two examples of time series forecasting. On the example of artificially generated time series, and on the example of a real one.


```python
def generate_synthetic_data(length: int = 2000, periods: int = 10):
    """
    The function generates a synthetic univariate time series

    :param length: the length of the array (even number)
    :param periods: the number of periods in the sine wave

    :return synthetic_data: an array without gaps
    """

    # First component
    sinusoidal_data = np.linspace(-periods * np.pi, periods * np.pi, length)
    sinusoidal_data = np.sin(sinusoidal_data)
    
    # Second component
    cos_1_data = np.linspace(-periods * np.pi/2, periods/2 * np.pi/2, int(length/2))
    cos_1_data = np.cos(cos_1_data) 
    cos_2_data = np.linspace(periods/2 * np.pi/2, periods * np.pi/2, int(length/2))
    cos_2_data = np.cos(cos_2_data)   
    cosine_data = np.hstack((cos_1_data, cos_2_data))
    
    random_noise = np.random.normal(loc=0.0, scale=0.1, size=length)

    # Combining a sine wave, cos wave and random noise
    synthetic_data = sinusoidal_data + cosine_data + random_noise
    return synthetic_data

# Get such numpy array
synthetic_time_series = generate_synthetic_data()

# We will predict 100 values in the future
len_forecast = 100

# Let's dividide our data on train and test samples
train_data = synthetic_time_series[:-len_forecast]
test_data = synthetic_time_series[-len_forecast:]

# Plot time series
plt.plot(np.arange(0, len(synthetic_time_series)), synthetic_time_series, label = 'Test')
plt.plot(np.arange(0, len(train_data)), train_data, label = 'Train')
plt.ylabel('Parameter', fontsize = 15)
plt.xlabel('Time index', fontsize = 15)
plt.legend(fontsize = 15)
plt.title('Synthetic time series')
plt.grid()
plt.show()
```


    
![png](3_intro_ts_forecasting_files/3_intro_ts_forecasting_11_0.png)
    


### Data preparation

FEDOT is a multi-functional framework that allows you to solve classification, regression, and clustering problems. Moreover, with FEDOT, you can use specific methods, such as time series forecasting. But to do this we need to get acquainted with two more entities:
* Task - the task that the pipeline will solve;
* InputData - data that is used for training and prediction.

Thus, to solve the problem, we must choose a model (or several models), put them in pipeline, define the Task as a time series forecasting, and put the input data in the InputData wrapper.

> Why do we need so much preparations?

The "pipeline" is a universal tool. In order for this universal tool to work well on a specific task, we have prepared a structure that helps the pipeline work effectively on a single task - we called it "Task". Here we will focus on the problem of time series forecasting - TaskTypesEnum.ts_forecasting.

FEDOT also has specific data preprocessing, so the data that is submitted to the input must be presented in a single format - "InputData".


```python
# Here we define which task should we use, here we also define two main forecast length
task = Task(TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=len_forecast))

# Prepare data to train the model
train_input = InputData(idx=np.arange(0, len(train_data)),
                        features=train_data,
                        target=train_data,
                        task=task,
                        data_type=DataTypesEnum.ts)

# Prepare input data for prediction part
start_forecast = len(train_data)
end_forecast = start_forecast + len_forecast
# Create forecast indices 
forecast_idx = np.arange(start_forecast, end_forecast)
predict_input = InputData(idx=forecast_idx,
                          features=train_data,
                          target=test_data,
                          task=task,
                          data_type=DataTypesEnum.ts)
```

Let's focus on parameters in more detail: forecast_length - the length of the sequence (in elements) that we want to make a forecast for. The "forecast_length" is just forecast of what length we want to build - it's clear, nothing more.

But we didn't make an important conversion - the lagged transformation. Without a tabular representation of the time series, we can't make a forecast using classical ML models. Such an operation in FEDOT is called 'lagged'. Let's create a pipeline with such data operation.


```python
# Create a pipeline
# lagged -> ridge 
node_lagged = PrimaryNode('lagged')
node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
ridge_pipeline = Pipeline(node_ridge)
```

Let's fit our pipeline and make our first forecast


```python
# Fit pipeline
ridge_pipeline.fit(train_input)

# Predict. Pipeline return OutputData object 
predicted_output = ridge_pipeline.predict(predict_input)

# Convert forecasted values into one-dimensional array
forecast = np.ravel(np.array(predicted_output.predict))
```


```python
def plot_results(actual_time_series, predicted_values, len_train_data, y_name = 'Parameter'):
    """
    Function for drawing plot with predictions
    
    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """
    
    plt.plot(np.arange(0, len(actual_time_series)), 
             actual_time_series, label = 'Actual values', c = 'green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)), 
             predicted_values, label = 'Predicted', c = 'blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c = 'black', linewidth = 1)
    plt.ylabel(y_name, fontsize = 15)
    plt.xlabel('Time index', fontsize = 15)
    plt.legend(fontsize = 15)
    plt.grid()
    plt.show()
```

Now let's use the function to draw graphs and look at the result


```python
plot_results(actual_time_series = synthetic_time_series,
             predicted_values = forecast, 
             len_train_data = len(train_data))
```


    
![png](3_intro_ts_forecasting_files/3_intro_ts_forecasting_20_0.png)
    


Not so good, but can we improve it? - Yes, we can. Below is description. 

Most operations in FEDOT have hyperparameters, and an important hyperparameter for lagged transformation is the size of the moving window (fig. 5).

<img src="../jupyter_media/time_series/window_sizes.png" width="900"/>

Figure 5. An example of the different window sizes

Below is an animation that shows how the entire process takes place from the formation of a training sample to the forecast for several values ahead (max_window_size = 3, forecast_length = 3):

<img src="../jupyter_media/time_series/animation_forecast.gif" width="900"/>

Thus, we can use different lags to make a forecast.

## Wrap-function for forecasting

Let's wrap our code into function, which will help us change some hyperparmeters. At the same time, let's look at what the whole process of building a model looks like.


```python
def make_forecast(train_data, len_forecast: int, window_size: int, final_model: str = 'ridge'):
    """
    Function for predicting values in a time series
    
    :param train_data: one-dimensional numpy array to train pipeline
    :param len_forecast: amount of values for predictions
    :param window_size: moving window size
    :param final_model: model in the root node

    :return predicted_values: numpy array, forecast of model
    """
    
    # Here we define which task should we use, here we also define two main forecast length
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Prepare input data for prediction part
    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast
    # Create forecast indices 
    forecast_idx = np.arange(start_forecast, end_forecast)
    predict_input = InputData(idx=forecast_idx,
                              features=train_data,
                              target=test_data,
                              task=task,
                              data_type=DataTypesEnum.ts)
    
    # Create a pipeline "lagged -> <final_model>" 
    node_lagged = PrimaryNode('lagged')
    
    # Define parameters to certain node 
    node_lagged.parameters = {'window_size': window_size}
    node_ridge = SecondaryNode(final_model, nodes_from=[node_lagged])
    ridge_pipeline = Pipeline(node_ridge)

    # Fit pipeline
    ridge_pipeline.fit(train_input)

    # Predict. Pipeline return OutputData object 
    predicted_output = ridge_pipeline.predict(predict_input)

    # Convert forecasted values into one-dimensional array
    forecast = np.ravel(np.array(predicted_output.predict))

    return forecast
```

Change the parameter of the lagged conversion to 400 values.


```python
predicted_values = make_forecast(train_data = train_data, 
                                 len_forecast = 100,
                                 window_size = 400)

plot_results(actual_time_series = synthetic_time_series,
             predicted_values = predicted_values, 
             len_train_data = len(train_data))
```


    
![png](3_intro_ts_forecasting_files/3_intro_ts_forecasting_25_0.png)
    


It looks good, but can Fedot handle real data? - Check it out!

## Real case. The height of the sea surface

Data was taken from a single cell of dataset "Sea level daily gridded data from satellite altimetry for the global ocean from 1993 to present". The full size dataset is available via [link](https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global?tab=overview). A time series with daily resolution of the sea surface height above the average surface of the geoid was prepared. Dataset contains 3 columns:
* Date;
* Level - sea surface height at the point with coordinates 10°N 120°W (WGS84), m;
* Neighboring level - sea surface height at the point with coordinates 10°N 121°W (WGS84), m.

### Load example data

Let's load the sea level data and try to predict the 250 next values.


```python
# Read the file
df = pd.read_csv('../data/ts_sea_level.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Specify forecast length
len_forecast = 250

# Got train, test parts, and the entire data
true_values = np.array(df['Level'])
train_array = true_values[:-len_forecast]
test_array = true_values[-len_forecast:]

plt.plot(df['Date'], true_values, label = 'Test')
plt.plot(df['Date'][:-len_forecast], train_array, label = 'Train')
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Sea level, m', fontsize = 15)
plt.legend(fontsize = 15)
plt.grid()
plt.show()
```


    
![png](3_intro_ts_forecasting_files/3_intro_ts_forecasting_28_0.png)
    


## Simple pipeline

Now let's prepare a pipeline of single model and try to predict the values


```python
# Make predictions
predicted_values = make_forecast(train_data=train_array, 
                                 len_forecast=250,
                                 window_size=400,
                                 final_model='ridge')
```


```python
# Plot predictions and true values
plot_results(actual_time_series=true_values,
             predicted_values=predicted_values, 
             len_train_data=len(train_array),
             y_name='Sea level, m')

# Print MAE metric
print(f'Mean absolute error: {mean_absolute_error(test_array, predicted_values):.3f}')
```


    
![png](3_intro_ts_forecasting_files/3_intro_ts_forecasting_31_0.png)
    


    Mean absolute error: 0.064
    

We can see, that on real data FEDOT also can show pretty good results.

### Complex pipeline

Let's set a complex pipeline of several branches (each branch with its own lagged transformation), see if we can achieve better results with more complex pipelines. Lets make an assumption that the source time series are multi-scale, so we will try to predict the sequence using two branches with different time lags. If you want to know more about multi-scale time series, follow this [notebook](4_auto_ts_forecasting.ipynb).


```python
# First level
node_lagged_1 = PrimaryNode('lagged')
node_lagged_1.parameters = {'window_size': 3}
node_lagged_2 = PrimaryNode('lagged')
node_lagged_2.parameters = {'window_size': 450}

# Second level
node_knnreg = SecondaryNode('knnreg', nodes_from=[node_lagged_1])
node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged_2])

# Third level - root node
node_final = SecondaryNode('ridge', nodes_from=[node_knnreg, node_ridge])
complex_pipeline = Pipeline(node_final)
```

Using the nodes_from argument, you can determine from which nodes the selected node will receive predictions. The prepared pipeline looks as follows

<img src="../jupyter_media/time_series/multi_lagged_chain.png" width="500"/>

Figure 6. 5-node pipeline for time series forecasting


```python
# Here we define which task should we use, here we also define two main forecast length
task = Task(TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=250))

# Prepare data to train the model
train_input = InputData(idx=np.arange(0, len(train_array)),
                        features=train_array,
                        target=train_array,
                        task=task,
                        data_type=DataTypesEnum.ts)

# Prepare input data for prediction part
start_forecast = len(train_array)
end_forecast = start_forecast + len_forecast
# Create forecast indices 
forecast_idx = np.arange(start_forecast, end_forecast)
predict_input = InputData(idx=forecast_idx,
                          features=train_array,
                          target=None,
                          task=task,
                          data_type=DataTypesEnum.ts)

# Fit pipeline
complex_pipeline.fit(train_input)

# Predict. Pipeline return OutputData object 
predicted_output = complex_pipeline.predict(predict_input)

# Convert forecasted values into one-dimensional array
predicted_values = np.ravel(np.array(predicted_output.predict))

# Plot predictions and true values
plot_results(actual_time_series = true_values,
             predicted_values = predicted_values, 
             len_train_data = len(train_array),
             y_name = 'Sea level, m')

# Print MAE metric
print(f'Mean absolute error: {mean_absolute_error(test_array, predicted_values):.3f}')
```


    
![png](3_intro_ts_forecasting_files/3_intro_ts_forecasting_35_0.png)
    


    Mean absolute error: 0.058
    

The error has become smaller! But we can improve our composite model one more time.

## Exogenous variables

We still have the time series forecasting task, but we want to use additional information. For example, sea level data from a nearby point. We can do it.


```python
# Plot two time series
plt.plot(df['Date'], df['Level'], label = 'sea height from point with coordinates 10°N 120°W')
plt.plot(df['Date'], df['Neighboring level'], label = 'sea height from point with coordinates 10°N 121°W')
plt.xlabel('Date', fontsize = 15)
plt.ylabel('Sea level, m', fontsize = 15)
plt.legend(fontsize = 15)
plt.grid()
plt.show()
```


    
![png](3_intro_ts_forecasting_files/3_intro_ts_forecasting_37_0.png)
    


Let's remember all the important aspects before training the model


```python
# Specify forecast length
len_forecast = 250
task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

# Got train, test parts, and the entire data
true_values = np.array(df['Level'])
train_array = true_values[:-len_forecast]
test_array = true_values[-len_forecast:]

# Data for lagged transformation
train_lagged = InputData(idx=np.arange(0, len(train_array)),
                         features=train_array,
                         target=train_array,
                         task=task,
                         data_type=DataTypesEnum.ts)
start_forecast = len(train_array)
end_forecast = start_forecast + len_forecast
predict_lagged = InputData(idx=np.arange(start_forecast, end_forecast),
                           features=train_array,
                           target=test_array,
                           task=task,
                           data_type=DataTypesEnum.ts)
```

Prepare an exogenous time series for train model and for prediction.


```python
exog_arr = np.array(df['Neighboring level'])
exog_train = exog_arr[:-len_forecast]
exog_test = exog_arr[-len_forecast:]

# Data for exog operation
train_exog = InputData(idx=np.arange(0, len(exog_train)),
                       features=exog_train,
                       target=train_array,
                       task=task,
                       data_type=DataTypesEnum.ts)
start_forecast = len(exog_train)
end_forecast = start_forecast + len_forecast
predict_exog = InputData(idx=np.arange(start_forecast, end_forecast),
                          features=exog_test,
                          target=test_array,
                          task=task,
                          data_type=DataTypesEnum.ts)
```

You can link different data to different nodes in the pipeline. To do this, there is a corresponding class - "MultiModalData", where you can match data by the name of the node.


```python
from fedot.core.data.multi_modal import MultiModalData

train_dataset = MultiModalData({
    'lagged': train_lagged,
    'exog_ts': train_exog
})

predict_dataset = MultiModalData({
    'lagged': predict_lagged,
    'exog_ts': predict_exog
})
```

Let's try using only a pipeline of one model to make a prediction more accurate using an exogenous variable than prediction from pipeline of models, but without using an exogenous time series.


```python
# Create a pipeline with different data sources in th nodes 
node_lagged = PrimaryNode('lagged')
node_exog = PrimaryNode('exog_ts')
node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged, node_exog])
exog_pipeline = Pipeline(node_ridge)

# Fit it
exog_pipeline.fit(train_dataset)

# Predict
predicted = exog_pipeline.predict(predict_dataset)
predicted_values = np.ravel(np.array(predicted.predict))
```


```python
# Plot predictions and true values
plot_results(actual_time_series = true_values,
             predicted_values = predicted_values, 
             len_train_data = len(train_array),
             y_name = 'Sea level, m')

# Print MAE metric
print(f'Mean absolute error: {mean_absolute_error(test_array, predicted_values):.3f}')
```

    Mean absolute error: 0.017
    

Thus, you can build pipelines of any configuration and complexity. Good luck!
