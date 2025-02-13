What is FEDOT
=============

FEDOT is an open-source framework for automated modeling and machine learning (AutoML). It produces a lightweight end-to-end ML solution in an automated way using an evolutionary approach.

FEDOT supports classification (binary and multiclass), regression, and time series forecasting tasks. FEDOT works both on unimodal (only tabular/image/text data) and multimodal data (more than one data source).

FEDOT supports a full life-сyсle of machine learning task that includes preprocessing, model selection, tuning, cross validation and serialization.

```python

    model = Fedot(problem='classification', timeout=5, preset='best_quality', n_jobs=-1)
    model.fit(features=x_train, target=y_train)
    prediction = model.predict(features=x_test)
    metrics = model.get_metrics(target=y_test)
```
