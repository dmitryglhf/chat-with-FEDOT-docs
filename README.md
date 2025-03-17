

<div align="center">

<img src="logo.png" alt="logo" width="50%"/>

## Chat with FEDOT docs

</div>

FEDOT is an open-source framework for automated modeling and machine learning (AutoML) problems. This framework is distributed under the 3-Clause BSD license.

It provides automatic generative design of machine learning pipelines for various real-world problems. The core of FEDOT is based on an evolutionary approach and supports classification (binary and multiclass), regression, clustering, and time series prediction problems.

## Link for Chat Bot

[chat-with-fedot-docs.app](https://chat-with-fedot-docs.streamlit.app/)

## Fork this repo for your own use

1. Go to [https://makersuite.google.com](https://aistudio.google.com/prompts/new_chat?hl=ru)
2. Click on the `+ Create new secret` key button.
3. Use [Streamlit Community Cloud's secrets management feature](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) to add your API key via the web interface. Add the following to it `genai_key=<your key here>`.

***

# FEDOT cheat sheet
[click me to download](https://github.com/dmitryglhf/chat-with-FEDOT-docs/blob/main/fedot-cheatsheet.pdf)

## Quick Start
```python
# Installation
pip install fedot

# Basic usage example
from fedot.api.main import Fedot
model = Fedot(problem='classification')
model.fit(features=X_train, target=y_train)
prediction = model.predict(features=X_test)
```

## Core Concepts
- **Pipelines**: Computational graphs that combine data operations and models
- **Nodes**: Individual operations within a pipeline (preprocessing or models)
- **Edges**: Connections between nodes that define data flow
- **Composer**: Algorithm that creates pipeline structures
- **Tuner**: Optimizes hyperparameters of models in a pipeline

## How does FEDOT pipeline works:

For example we had pipeline like this:
[some_pipeline](https://github.com/dmitryglhf/chat-with-FEDOT-docs/blob/main/pipeline.png)

How Prediction Works:

1. **First Level:** Initially, feature scaling is performed before feeding the data into each of the nodes.
2. **Second Level:** Each node sequentially predicts class probabilities. For example, in the s4e6 Kaggle example, there are three classes. Since there are three models at this level, the output is a matrix with nine columns (three probabilities for each model). The first three columns represent class probabilities from CatBoost, the next three are probabilities from XGBoost, and the final three columns are probabilities from LightGBM.
3. **Third Level:** At this stage, the data passed to the linear model (acting as a meta-model) consists of the predictions from each boosting model, rather than the original features.
4. **Final Prediction:** The linear model, as the meta-model, makes the final prediction based on the predictions of the previous models.

You can image it like this:
[how_some_pipeline_works](https://github.com/dmitryglhf/chat-with-FEDOT-docs/blob/main/pipeline_work.png)

## Common Methods

### Automatic Pipeline Construction
| Method | Description | Example |
|--------|-------------|---------|
| `Fedot()` | Initialize the framework | `model = Fedot(problem='regression')` |
| `fit()` | Train the model | `model.fit(features=X, target=y)` |
| `predict()` | Make predictions | `predictions = model.predict(features=X_test)` |
| `predict_proba()` | Predict probabilities | `probas = model.predict_proba(features=X_test)` |
| `save()` | Save the model | `model.save('model.pkl')` |
| `load()` | Load a saved model | `model = Fedot.load('model.pkl')` |

### Manual Pipeline Construction
| Method | Description | Example |
|--------|-------------|---------|
| `Pipeline()` | Create empty pipeline | `pipeline = Pipeline()` |
| `Node()` | Create operation node | `node = Node(operation_type='xgboost')` |
| `add_node()` | Add node to pipeline | `pipeline.add_node(node)` |
| `add_edge()` | Connect nodes | `pipeline.add_edge(node_from, node_to)` |
| `fit()` | Train pipeline | `pipeline.fit(input_data=train_data)` |
| `predict()` | Get predictions | `predictions = pipeline.predict(input_data=test_data)` |

## Configuration Options
```python
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


# Configuring FEDOT
model = Fedot(
    problem='classification',
    preset='best_quality',  # Options: 'fast', 'stable', 'best_quality', etc.
    timeout=5,  # Minutes for optimization
    with_tuning=True,  # Allow tuning mode
    n_jobs=-1,  # CPU cores to use (-1 = all)
    cv_folds=5,  # Cross-validation folds
    seed=42,  # Random seed
    metric=["accuracy"],  # Set metrics to optimize
    initial_assumption=PipelineBuilder() \
    .add_node('catboost', params={"iterations": 10000}).build(),  # Set new initial assumption
)

```

## Common Patterns / Recipes

### Time Series Forecasting
```python
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup

# Time series forecasting
model = Fedot(problem='ts_forecasting', 
              forecast_length=10)  # Predict 10 steps ahead

# Prepare time series data
dataframe = pd.read_csv('timeseries_data.csv')
train_input, test_input = train_test_data_setup(dataframe)

# Fit and forecast
model.fit(train_input)
forecast = model.predict(test_input)
```

### Multi-Modal Data Processing
```python
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PipelineNode
from fedot.core.data.multi_modal import MultiModalData

# Create nodes for different data types
text_node = PipelineNode('text_clean')
tabular_node = PipelineNode('scaling')
output_node = PipelineNode('rf')

# Create pipeline
pipeline = Pipeline()
pipeline.add_node(text_node)
pipeline.add_node(tabular_node)
pipeline.add_node(output_node)

# Connect nodes
pipeline.add_edge(text_node, output_node)
pipeline.add_edge(tabular_node, output_node)

# Create multi-modal data
data = MultiModalData({
    'text': text_data,
    'tabular': tabular_data
})

# Fit and predict
pipeline.fit(data)
prediction = pipeline.predict(test_data)
```

## Available Models and Operations

### Primary Nodes (Data Processing)
- `scaling` - Feature scaling
- `normalization` - Feature normalization
- `pca` - Principal Component Analysis
- `kernel_pca` - Kernel PCA
- `fast_ica` - Independent Component Analysis
- `polynomial_features` - Polynomial feature generation
- `lagged` - Time series lagged features
- `smoothing` - Time series smoothing
- etc.

### Secondary Nodes (Models)
- `linear` - Linear models
- `ridge` - Ridge regression
- `lasso` - Lasso regression
- `rf` - Random Forest
- `xgboost` - XGBoost
- `lgbm` - LightGBM
- `catboost` - CatBoost
- `knn` - K-Nearest Neighbors
- `dt` - Decision Tree
- `mlp` - Multi-layer Perceptron
- etc.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `MemoryError` during pipeline optimization | Reduce `pop_size` and `num_of_generations` parameters |
| Poor performance on time series | Add lagged features with `lagged` operation nodes |
| Slow execution | Set `preset='fast'` or reduce optimization parameters |
| Overfitting on small datasets | Use cross-validation (`cv_folds` parameter) or simpler pipelines |
| Model not improving | Try different `metric` values or increase optimization time |

## Resources
- [Official Documentation](https://fedot.readthedocs.io/)
- [GitHub Repository](https://github.com/aimclib/FEDOT)
- [Examples](https://github.com/ITMO-NSS-team/fedot-examples)
