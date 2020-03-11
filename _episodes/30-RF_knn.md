---
title: "Random forest regression. K nearest neighbor regression"
author: "Madhura Killedar, Darya Vanichkina"

keypoints: 
- Random forests can be combined to solve regression tasks
- kNN is a method that can also be used for regression
objectives:
- To fit a RF and KNN model to our data
- To explore the effect of hyperparameters on model fit
questions:
- How do we implement tree-based and proximity-based methods in python?
source: Rmd
teaching: 45
exercises: 45
bibliography: references.bib
---



## Random forest regression. K nearest neighbor regression


```python
websiterendering = True
# when delivering live coding, these libraries and code in this cell have already been loaded
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import statsmodels.api as sm
import seaborn as sns
import pickle

import sys
sys.path.insert(0, 'py-earth')
from pyearth import Earth

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.utils import resample


# Set up plotting options for seaborn and matplotlib
sns.set_context('notebook') 
sns.set_style('ticks') 
%matplotlib inline
plt.rcParams['figure.figsize'] = (9, 6)

# load from previous lessons
cached_files = ['models/ames_train_y.pickle','models/ames_test_y.pickle',
                'models/ames_train_X.pickle','models/ames_test_X.pickle',
                'models/predictors.pickle','models/ames_ols_all.pickle',
                'models/ames_ridge.pickle','models/ames_lasso.pickle', 
                'models/ames_enet.pickle','models/ames_mars.pickle',
               'models/ames_pcr.pickle', 'models/ames_plsr.pickle']

for file in cached_files:
    with open(file, 'rb') as f:
        objectname = file.replace('models/', '').replace('.pickle', '')
        exec(objectname + " = pickle.load(f)")
        f.close()

## 
def assess_model_fit(models,
                     model_labels, 
                     datasetX, 
                     datasetY):
    columns = ['RMSE', 'R2', 'MAE']
    rows = model_labels
    results = pd.DataFrame(0.0, columns=columns, index=rows)
    for i, method in enumerate(models):
        tmp_dataset_X = datasetX
        # while we build the model and predict on the log10Transformed 
        # sale price, we display the error in dollars as that makes more sense
        y_pred=10**(method.predict(tmp_dataset_X))
        results.iloc[i,0] = np.sqrt(mean_squared_error(10**(datasetY), y_pred))
        results.iloc[i,1] = r2_score(10**(datasetY), y_pred)
        results.iloc[i,2] = mean_absolute_error(10**(datasetY), y_pred)
    return(results.round(3))
```

## Random Forest
In random forest, each tree in the ensemble is built from a bootstrap sample from the training set. In addition, when splitting a node during the construction of the tree, the split that is chosen is the best split among a random subset of the features.


```python
# tuning grid was defined to optimise the following RF parameters:
param_grid = {"n_estimators": list(np.arange(10,160,10)),
            'max_depth': list(np.arange(3,11,1)),
            'min_samples_split': [0.005, 0.01, 0.02],
             'max_features': ['sqrt', 'auto']}
```

This was optimised on the HPC (we'll see some sample scripts for this in the next session), and the best outcome of this ended up being:

```
{'max_depth': 9, 'min_samples_split': 0.005, 
'max_features': 'auto', 'n_estimators': 150}

# best score
0.8735794018428228
```



```python
from sklearn.ensemble import RandomForestRegressor

ames_RF = Pipeline([
    ('estimator', RandomForestRegressor(n_estimators=150, 
                                       max_depth = 9,
                                       min_samples_split = 0.005,
                                       max_features = 'auto'))
    # If we want to actually tune these parameters
    #('estimator', GridSearchCV(RandomForestRegressor(), param_grid, scoring='r2', cv=10))
])

if websiterendering:
    with open('models/ames_RF.pickle', 'rb') as f:
        ames_RF = pickle.load(f)
else:
    # STUDENTS: execute the line of code below
    ames_RF.fit(ames_train_X, ames_train_y)
    pickle.dump(ames_RF, open('models/ames_RF.pickle', 'wb'))
    
#best_RF = ames_RF.named_steps.estimator.best_estimator_
#print(best_RF)
```


```python
ames_RF.named_steps['estimator']
```




    RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                          max_depth=9, max_features='auto', max_leaf_nodes=None,
                          max_samples=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=0.005, min_weight_fraction_leaf=0.0,
                          n_estimators=150, n_jobs=None, oob_score=False,
                          random_state=None, verbose=0, warm_start=False)



> ## Challenge 1
>
> 1. Try different hyperparameters, how does it impact the feature importances and RMSE (see below)?
> 
> {: .source}
>
{: .challenge}


```python
def plot_coefficients(model, labels):
    table = pd.Series(model.feature_importances_, index = labels)
    # Get the largest 20 values (by absolute value)
    table = table[table.abs().nlargest(20).index].sort_values()

    fig, ax = fig, ax = plt.subplots()
    table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    ax.set_title('Feature Importances (twenty largest in absolute value)', fontsize=14)
    sns.despine()
    return fig, ax
```


```python
plot_coefficients(ames_RF.named_steps['estimator'], predictors)
plt.show()
```


![png](../fig/30-RF_knn_9_0.png)


## k-Nearest Neighbours Regression


```python
# tuning grid will be defined to optimise the following knn parameters:
param_grid = {"n_neighbors": list(np.arange(3,21,2)),
              "weights": ['uniform','distance'],
             }

# Results from tuning:
# print(ames_kNN.named_steps['estimator'].best_score_)
# 0.7842456772785913
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=7, p=2, weights='distance')
```


```python
from sklearn.neighbors import KNeighborsRegressor

# Next, let's try to tune locally, trying 6, 7 and 8 neighbors:
param_grid = {"n_neighbors": [6,7,8],
              "weights": ['uniform']}


ames_kNN = Pipeline([
    ('scaler', StandardScaler()),
    #('scaler', RobustScaler()),
    #('estimator', KNeighborsRegressor(n_neighbors=10))
    ('estimator', GridSearchCV(KNeighborsRegressor(), 
                               param_grid, scoring='r2', cv=10))
])


if websiterendering:
    with open('models/ames_knn.pickle', 'rb') as f:
        ames_kNN = pickle.load(f)
else:
    # STUDENTS: RUN THE LINE BELOW ONLY:
    ames_kNN.fit(ames_train_X, ames_train_y)
    pickle.dump(ames_kNN, open('models/ames_knn.pickle', 'wb'))

    
    
print(ames_kNN.named_steps['estimator'].best_estimator_)
print(ames_kNN.named_steps['estimator'].best_score_)
```

    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=None, n_neighbors=8, p=2,
                        weights='uniform')
    0.7775511840741828


## Compare Models


```python
# What was the RMSE on the training data?
assess_model_fit(models=[ames_ols_all, ames_ridge, ames_lasso, ames_enet, 
                         ames_pcr, ames_plsr,ames_mars, ames_RF, ames_kNN],
                 model_labels=['OLS','Ridge', 'Lasso', 'ENet','PCR','PLSR', 'MARS','RF', 'kNN'],
                 datasetX=ames_train_X,
                 datasetY=ames_train_y).sort_values("RMSE")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RMSE</th>
      <th>R2</th>
      <th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OLS</th>
      <td>15757.714</td>
      <td>0.961</td>
      <td>10931.668</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>16208.353</td>
      <td>0.958</td>
      <td>11033.811</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>16480.809</td>
      <td>0.957</td>
      <td>11448.829</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>16497.181</td>
      <td>0.957</td>
      <td>11462.724</td>
    </tr>
    <tr>
      <th>PLSR</th>
      <td>16524.567</td>
      <td>0.957</td>
      <td>11602.496</td>
    </tr>
    <tr>
      <th>PCR</th>
      <td>16764.468</td>
      <td>0.955</td>
      <td>11776.527</td>
    </tr>
    <tr>
      <th>ENet</th>
      <td>17041.271</td>
      <td>0.954</td>
      <td>11799.798</td>
    </tr>
    <tr>
      <th>MARS</th>
      <td>19172.923</td>
      <td>0.942</td>
      <td>13498.476</td>
    </tr>
    <tr>
      <th>kNN</th>
      <td>31651.974</td>
      <td>0.841</td>
      <td>20155.938</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What was the RMSE on the testing data?
assess_model_fit(models=[ames_ols_all, ames_ridge, ames_lasso, ames_enet, 
                         ames_pcr, ames_plsr, ames_mars, ames_RF, ames_kNN],
                 model_labels=['OLS','Ridge', 'Lasso', 'ENet','PCR','PLSR','MARS', 'RF', 'kNN'],
                 datasetX=ames_test_X,
                 datasetY=ames_test_y).sort_values("RMSE")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RMSE</th>
      <th>R2</th>
      <th>MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ENet</th>
      <td>19801.125</td>
      <td>0.933</td>
      <td>13317.465</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>19864.493</td>
      <td>0.933</td>
      <td>13120.146</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>20024.975</td>
      <td>0.932</td>
      <td>13270.709</td>
    </tr>
    <tr>
      <th>PLSR</th>
      <td>20113.237</td>
      <td>0.931</td>
      <td>13372.746</td>
    </tr>
    <tr>
      <th>OLS</th>
      <td>20541.485</td>
      <td>0.928</td>
      <td>13346.733</td>
    </tr>
    <tr>
      <th>PCR</th>
      <td>21165.295</td>
      <td>0.924</td>
      <td>14049.317</td>
    </tr>
    <tr>
      <th>MARS</th>
      <td>23226.020</td>
      <td>0.908</td>
      <td>15355.804</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>27320.009</td>
      <td>0.873</td>
      <td>16817.876</td>
    </tr>
    <tr>
      <th>kNN</th>
      <td>34498.941</td>
      <td>0.797</td>
      <td>22983.686</td>
    </tr>
  </tbody>
</table>
</div>


