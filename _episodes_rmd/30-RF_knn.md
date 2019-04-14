---
title: "Random forest regression. K nearest neighbor regression"
author: "Madhura Killedar"
exercises: 30
keypoints: 
- Random forests can be combined to solve regression tasks
- kNN is a method that can also be used for regression
source: Rmd
start: 0
teaching: 30
bibliography: references.bib
---



## Random forest regression. K nearest neighbor regression


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import statsmodels.api as sm
import seaborn as sns
import pickle

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
```


```python
# load from previous lessons
cached_files = ['models/ames_train_y.pickle','models/ames_test_y.pickle',
                'models/ames_train_X.pickle','models/ames_test_X.pickle',
                'models/predictors.pickle','models/ames_ols_all.pickle',
                'models/ames_ridge.pickle','models/ames_lasso.pickle', 
                'models/ames_enet.pickle']

for file in cached_files:
    with open(file, 'rb') as f:
        objectname = file.replace('models/', '').replace('.pickle', '')
        exec(objectname + " = pickle.load(f)")
        f.close()
```

## Random Forest
In random forest, each tree in the ensemble is built from a bootstrap sample from the training set. In addition, when splitting a node during the construction of the tree, the split that is chosen is the best split among a random subset of the features. The scikit-learn implementation combines classifiers by averaging their probabilistic prediction, instead of letting each classifier vote for a single class.


```python
# tuning grid was defined to optimise the following RF parameters:
param_grid = {"n_estimators": list(np.arange(10,160,10)),
            'max_depth': list(np.arange(3,11,1)),
            'min_samples_split': [0.005, 0.01, 0.02],
             'max_features': ['sqrt', 'auto']}

```

This was optimised on the HPC (we'll see some sample scripts for this in the next session), and the best outcome of this ended up being:

```
{'max_depth': 10, 'n_estimators': 140, 
'max_features': 'auto', 'min_samples_split': 2}

# best score
0.8770534214301398 
```



```python
from sklearn.ensemble import RandomForestRegressor

ames_RF = Pipeline([
    ('estimator', RandomForestRegressor(n_estimators=140, 
                                       max_depth = 10,
                                       min_samples_split = 2,
                                       max_features = 'auto'))
    #('estimator', GridSearchCV(RandomForestRegressor(), param_grid, scoring='r2', cv=10))
])



## Toggle comment below to build model
#ames_RF.fit(ames_train_X, ames_train_y)
#pickle.dump(ames_RF, open('models/ames_rforest.pickle', 'wb'))
with open('models/ames_rforest.pickle', 'rb') as f:
    ames_RF = pickle.load(f)
```


```python
best_RF = ames_RF.named_steps['estimator']
print(best_RF)
```

    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=140, n_jobs=None,
               oob_score=False, random_state=None, verbose=0, warm_start=False)


> ## Challenge 1
>
> 1. Look at the coefficients for the model above. How many trees are worth combining?
> 2. Which minimum leaf size is best?
> 
> {: .source}
>
> > ## Solution
> > 
> > 1.
> > 2. 
> > {: .output}
> {: .solution}
{: .challenge}

> ## Challenge 2
>
> Explore values of n_estimators from 80 to 150 in increments of 1 (as a group. Don't forget to set random_state to be 42!). 
> Who is able to get the best results? Are they better than the above?
> 
> {: .source}
>
> > ## Solution
> > 
> > ~~~
> > 
> > param_test = {'n_estimators': [V1, V2, V3],
> >             'max_depth': [10],
> >             'min_samples_split': [2],
> >             'max_features': ['sqrt']}
> > 
> > 
> > ames_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
> >    param_grid=param_test,
> >    n_jobs=4,
> >    iid=False,
> >    cv=10)
> > 
> > ames_rf.fit(ames_train_X, ames_train_y)
> > 
> > print(ames_rf.best_score_,"\t",ames_rf.best_params_)
> > 
> > ~~~
> > 
> > {: .output}
> {: .solution}
{: .challenge}


```python
def plot_coefficients(model, labels):
    importance = model.feature_importances_

    table = pd.Series(importance.ravel(), index = labels).sort_values(ascending=True, inplace=False)
    
    reference = pd.Series(np.abs(importance.ravel()), index = labels).sort_values(ascending=False, inplace=False)
    reference = reference.iloc[:20]
    table = table[reference.index]
    table = table.sort_values(ascending=True, inplace=False)

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


![png](../fig/30-RF_knn_11_0.png)


## k-Nearest Neighbours Regression


```python
# tuning grid will be defined to optimise the following knn parameters:
param_grid = {"n_neighbors": list(np.arange(3,21,2)),
              "weights": ['uniform','distance'],
             }

# print(ames_kNN.named_steps['estimator'].best_score_)
#0.7842456772785913
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
    ('estimator', GridSearchCV(KNeighborsRegressor(), param_grid, scoring='r2', cv=10))
])


## Toggle comment below to build model
#ames_kNN.fit(ames_train_X, ames_train_y)
#pickle.dump(ames_kNN, open('models/ames_knn.pickle', 'wb'))
with open('models/ames_knn.pickle', 'rb') as f:
    ames_kNN = pickle.load(f)

print(ames_kNN.named_steps['estimator'].best_estimator_)
print(ames_kNN.named_steps['estimator'].best_score_)
```

    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=None, n_neighbors=7, p=2,
              weights='uniform')
    0.7810186506203685


## Compare Models


```python
# What was the RMSE on the training data?
columns=['Train RMSE']
rows=['OLS','Ridge', 'Lasso', 'ENet', 'Random Forest', 'k Nearest Neighbours']
results=pd.DataFrame(0.0, columns=columns, index=rows) 

methods=[ames_ols_all, ames_ridge, ames_lasso, ames_enet, ames_RF, ames_kNN]

for i, method in enumerate(methods):
    y_pred=method.predict(ames_train_X)
    results.iloc[i,0] = np.sqrt(mean_squared_error(10**ames_train_y, 10**y_pred))

results.round(2)
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
      <th>Train RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OLS</th>
      <td>18810.89</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>21330.85</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>20365.80</td>
    </tr>
    <tr>
      <th>ENet</th>
      <td>21292.16</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>12635.18</td>
    </tr>
    <tr>
      <th>k Nearest Neighbours</th>
      <td>32778.02</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compare with the test data!
columns=['Test RMSE']
rows=['OLS','Ridge', 'Lasso', 'ENet', 'Random Forest', 'k Nearest Neighbours']
results=pd.DataFrame(0.0, columns=columns, index=rows) 

methods=[ames_ols_all,  ames_ridge, ames_lasso, ames_enet, ames_RF, ames_kNN]

for i, method in enumerate(methods):
    y_pred=method.predict(ames_test_X)
    results.iloc[i,0] = np.sqrt(mean_squared_error(10**ames_test_y, 10**y_pred))

results.round(2)
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
      <th>Test RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OLS</th>
      <td>64792.91</td>
    </tr>
    <tr>
      <th>Ridge</th>
      <td>47670.17</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>59592.17</td>
    </tr>
    <tr>
      <th>ENet</th>
      <td>52482.81</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>27149.50</td>
    </tr>
    <tr>
      <th>k Nearest Neighbours</th>
      <td>36781.78</td>
    </tr>
  </tbody>
</table>
</div>


