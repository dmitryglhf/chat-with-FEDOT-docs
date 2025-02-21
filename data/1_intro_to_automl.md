# AutoML solution vs single model
#### FEDOT version = 0.7.2


```python
!pip install fedot==0.7.2
```

Below is an example of running an Auto ML solution for a classification problem.
## Description of the task and dataset


```python
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import logging
logging.raiseExceptions = False

# Input data from csv files 
train_data_path = '../data/scoring_train.csv'
test_data_path = '../data/scoring_test.csv'
df = pd.read_csv(train_data_path)
df.head(5)
```

## Baseline model

Let's use the api features to solve the classification problem. First, we create a pipeline with a single model "xgboost". 
To do this, we will substitute the appropriate name in the predefined_model field.

Attention!
"predefined_model" - is not an initial assumption for the AutoML algorithm. It's just a single model without AutoML part


```python
from fedot.api.main import Fedot

# task selection, initialisation of the framework
baseline_model = Fedot(problem='classification')

# fit model without optimisation - single XGBoost node is used 
xgb_pipeline = baseline_model.fit(features=train_data_path, target='target', predefined_model='xgboost')

# evaluate the prediction with test data
xgb_predict = baseline_model.predict_proba(features=test_data_path)
```

    2023-10-02 19:07:00,226 - CSV data extraction - Used the column as index: "ID".
    2023-10-02 19:07:10,038 - FEDOT logger - Final pipeline: {'depth': 1, 'length': 1, 'nodes': [xgboost]}
    xgboost - {'eval_metric': 'mlogloss', 'nthread': -1}
    2023-10-02 19:07:10,041 - MemoryAnalytics - Memory consumption for finish in main session: current 35.9 MiB, max: 63.5 MiB
    2023-10-02 19:07:10,114 - CSV data extraction - Used the column as index: "ID".
    


```python
from fedot.core.data.data import InputData
from sklearn.metrics import roc_auc_score

# Read data from csv file as InputData
test_data = InputData.from_csv(test_data_path)
roc_auc_baseline = roc_auc_score(test_data.target, xgb_predict)
```



## FEDOT AutoML for classification

We can identify the model using an evolutionary algorithm built into the core of the FEDOT framework.

Here are some parameters that you can specify when initializing a class:
* problem - the name of modelling problem to solve:
        - classification
        - regression
        - ts_forecasting
        - clustering
* seed - value for fixed random seed
* logging_level - level of the output detailing
        - 50 -> critical
        - 40 -> error
        - 30 -> warning
        - 20 -> info
        - 10 -> debug
        - 0 -> nonset
* timeout - time for model design (in minutes)


```python
# new instance to be used as AutoML tool
auto_model = Fedot(problem='classification', seed=42, logging_level=10, timeout=5)
```


```python
# run of the AutoML-based model generation
pipeline = auto_model.fit(features=train_data_path, target='target')
```

    



```python
# comparison with the manual pipeline

print(f'Baseline {roc_auc_baseline:.2f}')
print(f'AutoML solution {roc_auc_auto:.2f}')
```

    Baseline 0.83
    AutoML solution 0.85
    

Thus, with just a few lines of code, we were able to launch the FEDOT framework and got a better result*.

*Due to the stochastic nature of the algorithm, the metrics for the found solution may differ.
