Tabular Data Prediction
==============================================

Introduction

As common AutoML frameworks, FEDOT solves problems with data that are represented as tables.
FEDOT allows you to automate machine learning pipeline design for tabular data in ``classification`` and ``regression``
problems.

Also, it provides a high-level API that enables you to use common fit/predict interface. To use API it is required
to import certain object:

```python

    from fedot.api.main import Fedot
```

Loading training and test data from a CSV file as a Pandas dataframe ``pd.DataFrame``.

```python

    train = pd.DataFrame('train.csv')
    test = pd.DataFrame('test.csv')
```

Initialize the FEDOT object and define the type of modeling problem. In this case, problem is ``classification``.

```python

    model = Fedot(problem='classification', metric='roc_auc')
```

.. note::

    Class ``Fedot.__init__()`` has more than two params, e.g. ``timeout`` for setting time limits or
    ``n_jobs`` for parallelization. For more details, see the :doc:`FEDOT API </api/api>` section in our documentation.

The ``fit()`` method begins the optimization and returns the resulting composite pipeline.

```python

    best_pipeline = model.fit(features=train, target='target')
```

After the fitting is completed, you can look at the structure of the resulting pipeline.
For example, let best pipeline consist of two nodes: resampling operation (*resample*) and Random Forest (*rf*).
Let see how it looks like.

In text format:

```python

    best_pipeline.print_structure()
```

```text

    Pipeline structure:
    {'depth': 2, 'length': 2, 'nodes': [rf, resample]}
    rf - {'n_jobs': -1, 'bootstrap': False, 'criterion': 'entropy', 'max_features': 0.2452946642710205, 'min_samples_leaf': 6, 'min_samples_split': 4, 'n_estimators': 100}
    resample - {'balance': 'expand_minority', 'replace': False, 'balance_ratio': 0.5984630982827773}
```

And in plot format:

```python

    best_pipeline.show()
```

The ``predict()`` method, which uses an already fitted pipeline, returns values for the target.

```python

    prediction = model.predict(features=test)
```

.. hint::

    If you want to predict target probability use ``predict_proba()`` method.

The ``get_metrics()`` method estimates the quality of predictions according the selected metrics.

```python

    prediction = model.get_metrics()
```

.. note::

    The same way FEDOT can be used to ``regression`` problem. It is only required to change params according the problem
    in main class object:

```python

    model = Fedot(problem='regression', metric='rmse')
```
