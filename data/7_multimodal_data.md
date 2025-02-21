<center><img src=https://github.com/nccr-itmo/FEDOT/raw/master/docs/fedot_logo.png></img></center>

# <center>Case of multimodal data classification using the [FEDOT](https://github.com/nccr-itmo/FEDOT) framework</center>

# Introduction

**FEDOT** is an open source automatic machine learning framework that is capable of automating the creation and optimization of machine learning pipelines and their elements.
The framework allows you to compactly and efficiently solve various modeling problems.

**Multimodal data** is data that has a different nature (tables, text, images, time series). Humans perceive the world in [a multimodal way](https://en.wikipedia.org/wiki/McGurk_effect), so using this approach in machine learning can also work. Indeed, the sharing of several types of data improves the quality of the model at the expense of information that may be contained in one modality and absent in another.

### FEDOT installation

Let's install FEDOT of version 0.7.2


```python
!pip install fedot==0.7.2
```

<div class="alert alert-block alert-info"><b>Note</b>: there are alternative ways to install the framework and its dependencies, which can be found in the section <b><a href="https://fedot.readthedocs.io/en/latest/introduction/tutorial/quickstart.html">Quickstart</a></b> in our documentation</div>

## Data loading

As an example, let's take the prepared data from the [Wine Reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews) dataset (winemag-data_first150k). For convenience and speed of work, the number of lines and classes has been reduced, no other preprocessing has been carried out (we will leave this to Fedot).

The **description** field contains a textual description of the different varieties of wine, and the target variable is the **variety** field.


```python
import pandas as pd
path = '../data/multimodal_wine.csv'

df = pd.read_csv(path)
df
```

Data has been loaded. Now you can try several solutions.

## Using FEDOT's multimodal functionality

You can solve this classification problem using FEDOT with a few lines of code. First, we import the necessary modules and classes, and also split the selection into train and test parts.

FEDOT will automatically detect the **description** field as a text field and apply various text processing models to it. You should also point out the column with the target variable **variety** and the absence of a separate index column in the data.


```python
from fedot.api.main import Fedot
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData

data = MultiModalData.from_csv(file_path=path, task='classification', target_columns='variety', index_col=None)
fit_data, predict_data = train_test_data_setup(data, shuffle_flag=True, split_ratio=0.7)
```

Let's set the parameters of the AutoML model.


```python
automl_model = Fedot(problem='classification', metric=['roc_auc', 'f1'], timeout=10, seed=42)
```

* `problem = 'classification'` - problem solved by the framework
* `timeout = 3` - framework time in minutes
* `seed = 42` - let's fix the seed for reproducibility

<div class="alert alert-block alert-info"><b>Note</b>: there are also other options for running FEDOT. More details can be found in the section <b><a href="https://fedot.readthedocs.io/en/latest/api/api.html">FEDOT API</a></b> in our documentation.</div>

Next, you can start the process of finding the optimal pipeline and its training. It is necessary to pass data for training to the `.fit()` method.


```python
automl_model.fit(features=fit_data,
                 target=fit_data.target)
```




Pipeline's structure in text form is shown above. For visualization, we will use the `.show()` method.


```python
automl_model.current_pipeline.show(node_size_scale=0.5, dpi=100)
```


    
![png](7_multimodal_data_files/7_multimodal_data_20_0.png)
    


Let's predict the target variable and compare the metrics.


```python
prediction = automl_model.predict(predict_data)

metrics = automl_model.get_metrics()

metrics
```




    {'roc_auc': 0.5, 'f1': 0.79}



Let's visualize the metric.


```python
automl_model.plot_prediction()
```


Let's save the history of the generation process in JSON format.


```python
automl_model.history.save('multimodal_history.json');
```

Using the history, you can look at the change in the value of the metrics for each stage of generation. FEDOT uses evolutionary algorithms and compares the metrics of several individual pipelines in each generation, so the best visualization method is a boxplot, where the middle line means the median value of the metric in a generation.


```python
automl_model.history.show.fitness_box(best_fraction=0.5, dpi=100)
```




You can also look at which specific models dominated the pipelines during the generation process.


```python
automl_model.history.show.operations_kde(dpi=100)
```


### Baseline

To compare the quality of the model, we will run FEDOT on the same data, but training only the standard initial pipeline.


```python
baseline_model = Fedot(problem='classification', metric=['roc_auc', 'f1'], seed=42)
baseline_model.fit(features=fit_data,
                 target=fit_data.target,
                 predefined_model='auto')
```




The `timeout` parameter is not used in this case, so we will remove it. The `predefined_model` parameter specifies that we will not run AutoML, but will simply train the model on the standard initial pipeline.

Let's visualize the pipeline.


```python
baseline_model.current_pipeline.show(node_size_scale=0.5, dpi=100)
```


    
![png](7_multimodal_data_files/7_multimodal_data_36_0.png)
    


Let's get and compare the metrics. It is noticeable that even a small training time significantly improves the quality metrics.


```python
prediction = baseline_model.predict(predict_data)

baseline_metrics = baseline_model.get_metrics()

baseline_metrics
```




    {'roc_auc': 0.5, 'f1': 0.681}



### Tabular data only

To verify the significance of both modalities, let's train the base model only on tabular data. To do this, simply exclude the **description** text field when loading the data.


```python
table_data = MultiModalData.from_csv(file_path=path, task='classification', target_columns='variety', 
                                     columns_to_drop=['description'], index_col=None)
table_fit_data, table_predict_data = train_test_data_setup(table_data, shuffle_flag=True, split_ratio=0.7)
```


```python
table_model = Fedot(problem='classification', metric=['roc_auc', 'f1'], seed=42)
table_model.fit(features=table_fit_data,
                 target=table_fit_data.target,
                 predefined_model='auto')
```


```python
table_model.current_pipeline.show(node_size_scale=0.5, dpi=100)
```


It is noticeable that the metrics of this model are noticeably worse than those of the multimodal one.


```python
prediction = table_model.predict(table_predict_data)

table_metrics = table_model.get_metrics()

table_metrics
```



### Metrics

Let's compare the metrics of the AutoML model, the multimodal baseline, and the tabular baseline.


```python
print(f'ROC-AUC of AutoML model is {round(metrics["roc_auc"], 3)}')
print(f'ROC-AUC of baseline model is {round(baseline_metrics["roc_auc"], 3)}')
print(f'ROC-AUC of baseline table model is {round(table_metrics["roc_auc"], 3)}')
```

    ROC-AUC of AutoML model is 0.5
    ROC-AUC of baseline model is 0.5
    ROC-AUC of baseline table model is 0.5
    


```python
print(f'F1 of AutoML model is {round(metrics["f1"], 3)}')
print(f'F1 of baseline model is {round(baseline_metrics["f1"], 3)}')
print(f'F1 of baseline table model is {round(table_metrics["f1"], 3)}')
```

    F1 of AutoML model is 0.79
    F1 of baseline model is 0.681
    F1 of baseline table model is 0.673
    

## Afterword

In this notebook, we showed you how to run the **FEDOT** framework to solve a classification problem using multimodal data using the API. As you can see, this is done quite simply.

Now you can try running the **FEDOT** automatic machine learning framework on your data.
