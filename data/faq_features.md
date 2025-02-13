Features
==========

How does FEDOT avoid data leakage and bias?
-------------------------------------------

    Before the usage of AutoML, the dataset is usually splitting to training and test part.
    Then, the cross-validation is used to estimate the value of fitness functions. The default number of folds is 5.
    It can be specified manually using  ``cv_folds`` parameter.

    To deal with potential bias, additional metrics can be passed to optimiser using ``metric`` parameter.
