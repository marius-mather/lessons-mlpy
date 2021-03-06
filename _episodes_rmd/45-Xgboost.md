---
title: "Gradient boosting. XGBoost"
author: "Darya Vanichkina"
exercises: 30
questions:
- What are some newer approaches to ML?
- What are their pros and cons?
objectives:
- To appreciate that more advanced methods can have a much larger number of parameters to optimise
- This optimisation is best done iteratively on the HPC
keypoints: 
- Modern approaches such as GBM and XGBoost can drastically improve prediction performance
- Tuning hyperparameters is, however, computationally expensive 
source: Rmd
teaching: 60
exercises: 30
bibliography: references.bib
---


### GBM. XGBoost


```python
# when delivering live coding, these libraries and all code in this cell have already been loaded
websiterendering = True
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
               'models/ames_pcr.pickle', 'models/ames_plsr.pickle',
               'models/ames_RF.pickle', 'models/ames_knn.pickle']

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


```python
# these need to be loaded for this section
import itertools
from sklearn.ensemble import GradientBoostingRegressor  # GBM algorithm
import xgboost as xgb
```

## Gradient boosting

Unlike algorithms we've worked with up to now, GBM has a substantially larger number of parameters. While we could optimise them all together using a GridSearchCV(), it makes more sense to do so iteratively, honing down on the best parameter sets one by one. 

First, let's take a `learning_rate` of 0.1, and identify how many `n_estimators` we'd use to fit the model. To do this:


```python
# param_test1 = {'n_estimators': list(range(20, len(predictors)+1, 10))}

# gbm1 = GridSearchCV(estimator=GradientBoostingRegressor(
#        loss='ls',
#        learning_rate=0.1, 
#        min_samples_split=20, 
#        min_samples_leaf=10,
#        max_depth=7, 
#        max_features='sqrt', 
#        subsample=0.8,   
#        random_state=42),
#    param_grid=param_test1, 
#    n_jobs=4, 
#    iid=False,
#    cv=10)
# gbm1.fit(ames_train_X, ames_train_y)
# pickle.dump(gbm1, open('models/gbm1.pickle', 'wb'))

with open('models/gbm1.pickle', 'rb') as f:
    gbm1 = pickle.load(f)

# print(gsearch1.grid_scores_)
print(gbm1.best_params_)
print(gbm1.best_score_)
```

    {'n_estimators': 280}
    0.9068897670500741



```python
# How many estimators did we test?
#print(param_test1)
#print(len(predictors))


# 
list(np.arange(0.001, 0.31, 0.05).round(2))
```




    [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]



We can see that the optimum number was found to be 280 (almost all of them!) - but we only tested 270 and 286. Perhaps the number is between these two? :et's see if we can find out:


```python
param_test1 = {'n_estimators': list(range(270, len(predictors)+1, 1))}

# gbm2 = GridSearchCV(estimator=GradientBoostingRegressor(
#        loss='ls',
#        learning_rate=0.1, 
#        min_samples_split=20, 
#        min_samples_leaf=10,
#        max_depth=7, 
#        max_features='sqrt', 
#        subsample=0.8,   
#        random_state=42),
#    param_grid=param_test1, 
#    n_jobs=4, 
#    iid=False, 
#    cv=10)
# gbm2.fit(ames_train_X, ames_train_y)


# print(gbm2.best_params_)
# print(gbm2.best_score_)
print(param_test1)
```

    {'n_estimators': [270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286]}


It turns out that with a `learning_rate` of 0.1, the optimal number of `n_estimators` is 280. (From this point, we start running things on the HPC, to achieve faster optimisation rates. However, this also means that we may not be able to match our pickles with those from our desktop, due to incompatibility of python package versions.

The next step is to optimise tree-specific parameters, starting with `max_depth` and `min_samples_split`. These correspond to:

- `min_samples_split` : The minimum number of samples required to split an internal node. Values around 0.5-1% of the dataset can be feasible
    - We will test values of 5 to 30, in increments of 5
- `max_depth` : maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. 
    - We will test values of 3 to 15, in increments of 2
    
This ends up being quite an expansive grid search, as we need to fit 6 * 7 = 42 models!


```python
param_test2 = {'min_samples_split': list(range(5, 31, 5)),
               'max_depth': list(range(3,16, 2))}
```

To do this, we can use Artemis, the university's HPC cluster. A simple way to do this is to fit each of the `min_samples_split` independently, using a python and bash script. The python script we'd use would look like this:
It would write out the best model to a tsv file, with the corresponding paramter, after which we could use Unix to look at them.

```
# Import libraries:
import os
os.chdir('/scratch/RDS-LALA/darya/gbm')

import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor  # GBM algorithm
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search



parser = argparse.ArgumentParser(description='GBM model fitting')
parser.add_argument('min_samples_split', type=int, help='min_samples_split value to test')
args = parser.parse_args()




# load from previous lessons
cached_files = ['models/ames_train_y.pickle','models/ames_test_y.pickle',
                'models/ames_train_X.pickle','models/ames_test_X.pickle',
                'models/predictors.pickle']

for file in cached_files:
    with open(file, 'rb') as f:
        objectname = file.replace('models/', '').replace('.pickle', '')
        exec(objectname + " = pickle.load(f)")
        f.close()




param_test = {'min_samples_split': [args.min_samples_split],
              'max_depth': list(range(3,16, 2))}

gbm2 = GridSearchCV(estimator=GradientBoostingRegressor(
       loss='ls',
       learning_rate=0.1,
       n_estimators = 279,
       min_samples_leaf=10,
       max_features='sqrt',
       subsample=0.8,
       random_state=42),
   param_grid=param_test,
   n_jobs=4,
   iid=False,
   cv=10)
gbm2.fit(ames_train_X, ames_train_y)

filename = 'fit1/' + str(args.min_samples_split) + 'fit.tsv'
with open(filename, 'w') as f:
    print(gbm2.best_score_,"\t",gbm2.best_params_, file = f)

```

The shell script we could use is very basic:

```
python /home/darya/gbm/optimise1.py $PARAM
```

And, finally, we could execute it using the following qsub commands:

```
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=5 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=10 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=15 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=20 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=25 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=30 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
```

The results for this are as follows. 
```
0.9107460935801089 	 {'max_depth': 7, 'min_samples_split': 30}
```

It's clear that we need to test higher values of the `min_samples_split`, so we can run another round of the optimisation with:



```
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=35 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=40 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=45 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=50 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
qsub -P RDS-CORE-SIH4HPC-RW -l select=1:ncpus=4:mem=4gb -l walltime=2:0:0 -N gbm1 -v PARAM=55 /home/dvanichkina/scratch_sih/darya/gbm/optimise1.sh
```

We then arrive at the following optimum:

```
0.9109466273291733 	 {'max_depth': 7, 'min_samples_split': 40}
```

To improve the fit of our model, let's try `max_depth` of 6, 7, or 8 and `min_samples_split` of 35 to 45 (increments of 1).
```
param_test = {'min_samples_split': list(range(35,46,1)),
              'max_depth': list(range(6,9, 1))}
```

We finally arrive at the optimum of:
```
0.9118552463171431 	 {'max_depth': 7, 'min_samples_split': 41}
```

***

Let's leave `max_depth` at 7, and test a range of values for:
- `min_samples_split` (35 to 45, increments of 1)
- `min_samples_leaf` (5 to 45, increments of 5)

We can use the HPC, as above, to run the code, or run it locally on our machine (takes ~20 mins):

```
param_test = {'min_samples_split': list(range(35,46,1)),
              'min_samples_leaf': list(range(5,36, 5))}

gbm2 = GridSearchCV(estimator=GradientBoostingRegressor(
       loss='ls',
       learning_rate=0.1, 
       n_estimators = 279,
       max_depth=7, 
       max_features='sqrt', 
       subsample=0.8,   
       random_state=42),
   param_grid=param_test, 
   n_jobs=4, 
   iid=False, 
   cv=10)
gbm2.fit(ames_train_X, ames_train_y)

print(gbm2.best_score_,"\t",gbm2.best_params_)
```


The outcome of this is

```
0.9151989049335256 	 {'min_samples_leaf': 5, 'min_samples_split': 36}
```

***

Next, let's tune the `max_features` parameter, trying values from 'auto' to 'log2' to 'sqrt' to only 20 parameters. 

```
param_test = {'max_features': ['auto','log2','sqrt'],
              'min_samples_leaf': list(range(5,21, 5))}

gbm2 = GridSearchCV(estimator=GradientBoostingRegressor(
       loss='ls',
       min_samples_split = 36,
       learning_rate=0.1, 
       n_estimators = 279,
       max_depth=7,  
       subsample=0.8,   
       random_state=42),
   param_grid=param_test, 
   n_jobs=4, 
   iid=False, 
   cv=10)
gbm2.fit(ames_train_X, ames_train_y)

print(gbm2.best_score_,"\t",gbm2.best_params_)
```

The resulf of this is:

`0.9151989049335256 	 {'min_samples_leaf': 5, 'max_features': 'sqrt'}`

***

Finally, let's go back to optimising the number of trees and learning rate, this time going with a lower learning rate (because the lower the learning rate, the slower the CV will be):



```
param_test = {'n_estimators': list(range(275, len(predictors)+1, 1)),
             'learning_rate': list(np.arange(0.001, 1.0, 0.1)),
             'subsample': list(np.arange(0.6, 1.0))}

gbm2 = GridSearchCV(estimator=GradientBoostingRegressor(
       loss='ls',
       max_depth=7, 
       max_features='sqrt', 
       min_samples_leaf=5,
       min_samples_split = 36,
       random_state=42),
   param_grid=param_test, 
   n_jobs=4, 
   iid=False, 
   cv=10)
gbm2.fit(ames_train_X, ames_train_y)

print(gbm2.best_score_,"\t",gbm2.best_params_)

```

`0.9117722351540429 	 {'learning_rate': 0.101, 'n_estimators': 280, 'subsample': 0.6}`

Note that we could also have "brute forced" the estimation using HPC, combining 3 scripts:

```
# optimise2.py ----------------
#
#
# Import libraries:
import os
os.chdir('/scratch/RDS-CORE-SIH4HPC-RW/darya/gbm')

import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor  # GBM algorithm
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search



parser = argparse.ArgumentParser(description='GBM model fitting')
parser.add_argument('min_samples_split', type=int, help='min_samples_split value to test')
parser.add_argument('n_estimators', type=int, help='n_estimators value to test')
args = parser.parse_args()


# load from previous lessons
cached_files = ['models/ames_train_y.pickle','models/ames_test_y.pickle',
                'models/ames_train_X.pickle','models/ames_test_X.pickle',
                'models/predictors.pickle']

for file in cached_files:
    with open(file, 'rb') as f:
        objectname = file.replace('models/', '').replace('.pickle', '')
        exec(objectname + " = pickle.load(f)")
        f.close()



# [args.min_samples_split]

param_test = {'n_estimators': [args.n_estimators], 
            'min_samples_split' : [args.min_samples_split], 
             'learning_rate': list(np.arange(0.001, 0.31, 0.05).round(2)), 
             'min_samples_leaf':  list(range(1,22, 4)),
             'max_depth': [1,3,5,6,7,8,9,10,15],
             'subsample': [0.65, 0.75, 0.8, 0.85, 0.9]
             }

gbm2 = GridSearchCV(estimator=GradientBoostingRegressor(
       loss='ls',
       max_features='sqrt',
       random_state=42),
   param_grid=param_test,
   n_jobs=4,
   iid=False,
   cv=10)


gbm2.fit(ames_train_X, ames_train_y)

filename = 'fit1/' + str(args.min_samples_split) + "_" + str(args.n_estimators) + 'fit.tsv'
with open(filename, 'w') as f:
    print(gbm2.best_score_,"\t",gbm2.best_params_, file = f)
    
    
    
    

####----
optimise2.pbs
#PBS -P PROJECTNAME
#PBS -N gbmOpt3
#PBS -l select=1:ncpus=3:mem=2gb
#PBS -l walltime=02:0:00
#PBS -J 100-286:5

# unload all modules
module purge

cd /home/dvanichkina/scratch_sih/darya/gbm/
python /home/dvanichkina/scratch_sih/darya/gbm/optimise2.py $PARAM $PBS_ARRAY_INDEX



#### --------------
# optimise2.sh
qsub -v PARAM=5 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=10 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=15 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=20 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=25 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=30 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=35 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=40 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs
qsub -v PARAM=45 /home/dvanichkina/scratch_sih/darya/gbm/optimise2.pbs

```

After all of this finishes running, the optimal parameter values we obtain are:
```
0.918410561940631 	 {'learning_rate': 0.051, 'n_estimators': 280, 'min_samples_split': 35, 'max_depth': 7, 'subsample': 0.8, 'min_samples_leaf': 1}
```

Let's fit this model locally:


```python
param_test = {'n_estimators': [280]}

ames_gbm = GridSearchCV(estimator=GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.051, 
        min_samples_split=35, 
        min_samples_leaf=1,
        max_depth=7, 
        max_features='sqrt', 
        subsample=0.8,   
        random_state=42),
    param_grid=param_test, 
    n_jobs=4, 
    iid=False,
    cv=10)

#ames_gbm.fit(ames_train_X, ames_train_y)
#pickle.dump(ames_gbm, open('models/ames_gbm.pickle', 'wb'))

with open('models/ames_gbm.pickle', 'rb') as f:
    ames_gbm = pickle.load(f)

# print(gsearch1.grid_scores_)
print(ames_gbm.best_params_)
print(ames_gbm.best_score_)
```

    {'n_estimators': 280}
    0.9182771422480472


***
## XGBoost


1. First, optimise `max_depth` and `min_child_weight`, with a varying number of `n_estimators` and a fixed, large `learning_rate` 0.1.


```
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn import cross_validation, metrics  # Additional scklearn functions
from sklearn.grid_search import GridSearchCV  # Perforing grid search
# load from previous lessons
cached_files = ['models/ames_train_y.pickle','models/ames_test_y.pickle',
                'models/ames_train_X.pickle','models/ames_test_X.pickle',
                'models/predictors.pickle']

for file in cached_files:
    with open(file, 'rb') as f:
        objectname = file.replace('models/', '').replace('.pickle', '')
        exec(objectname + " = pickle.load(f)")
        f.close()




param_test = {
 'max_depth':list(range(3,10,2)),
 'min_child_weight':list(range(1,7,2)),
 'n_estimators': [180, 230, 280]
}



xgbmodel = GridSearchCV(estimator=xgb.XGBRegressor(
  objective='reg:linear',
  learning_rate =0.1,
  n_estimators=280,
  gamma=0,
  subsample=0.8,
  colsample_bytree=0.8,
  random_state=42),
   param_grid=param_test,
   n_jobs=4,
   iid=False,
   cv=10)
   
xgbmodel.fit(ames_train_X, ames_train_y)
print(xgbmodel.best_score_,"\t",xgbmodel.best_params_)

# 0.9159701741062035 	 {'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 280}
```

Of these, it turns out the best parameter values are:
- 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 280

Now, let's see if we can improve this even more:

```
param_test = {
 'max_depth':[2,3,4],
 'min_child_weight':[0.5,1,1.5,2],
 'n_estimators': [275, 280, 285]
}

# 0.9162060722720288 	 {'max_depth': 3, 'min_child_weight': 0.5, 'n_estimators': 285}

```

2. Next, optimise gamma, the minimum split loss. This is the minimum loss reduction required to make a further partition on a leaf node of the tree. 

```
param_test = {
 'max_depth':[3],
 'min_child_weight':[0.5],
 'n_estimators': [283, 284, 285],
 'gamma': [i/10.0 for i in range(0,5)]
}

# 0.9162060722720288 	 {'gamma': 0.0, 'min_child_weight': 0.5, 'max_depth': 3, 'n_estimators': 285}
# Default zero seems best

```


3. Next, optimise `subsample` and `colsample_bytree`, which represent the proportion of data used to make each tree and how many features can go into a tree at each branch

```
param_test = {
 'max_depth':[3],
 'min_child_weight':[0.5],
 'n_estimators': [283, 284, 285, 286],
 'gamma': [0],
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]

}


# 0.916463171211643 	 {'max_depth': 3, 'gamma': 0, 'min_child_weight': 0.5, 'colsample_bytree': 0.6, 'subsample': 0.8, 'n_estimators': 284}

```

Let's zero in on that, and subsample values plus/minus 0.05 around the idenfied optima:

```
param_test = {
 'max_depth':[3],
 'min_child_weight':[0.5],
 'n_estimators': [283, 284, 285, 286],
 'gamma': [0],
 'subsample':[0.75, 0.775,  0.8, 0.825, 0.85],
 'colsample_bytree':[0.5, 0.55, 0.6]

}


# 0.918366305109824 	 {'max_depth': 3, 'gamma': 0, 'min_child_weight': 0.5, 'colsample_bytree': 0.5, 'subsample': 0.75, 'n_estimators': 285}

```


Perhaps need to reduce those two even more, as their values are the lowest that we're testing!

```

param_test = {
 'max_depth':[3],
 'min_child_weight':[0.5],
 'n_estimators': [283, 284, 285, 286],
 'gamma': [0],
 'subsample':[0.65, 0.7, 0.75, 0.8],
 'colsample_bytree':[0.4,0.45, 0.5, 0.55]

}
# 0.918366305109824 	 {'max_depth': 3, 'gamma': 0, 'min_child_weight': 0.5, 'colsample_bytree': 0.5, 'subsample': 0.75, 'n_estimators': 285}
```


4. Next, let's tune the regularisation parameters:

```
param_test = {
 'max_depth':[3],
 'min_child_weight':[0.5],
 'n_estimators': [285],
 'gamma': [0],
 'subsample':[0.75],
 'colsample_bytree':[0.5],
 'reg_alpha':[0, 0.1, 0.2, 0.5, 1],
 'reg_lambda': [0, 0.1, 0.2, 0.5, 1]
}

# 0.918366305109824 	 {'max_depth': 3, 'reg_lambda': 1, 'reg_alpha': 0, 'gamma': 0, 'min_child_weight': 0.5, 'colsample_bytree': 0.5, 'subsample': 0.75, 'n_estimators': 285}

# defaults we've been using so far: alpha = 0 , lambda = 1, are optimal for us (i.e. ridge regression-like)
```


Let's fit the final model, and assess its performance on the training and test sets:


```python
param_test = {
 'max_depth':[3],
 'min_child_weight':[0.5],
 'n_estimators': [285],
 'gamma': [0],
 'subsample':[0.75],
 'colsample_bytree':[0.5]
}

if websiterendering:
    with open('models/ames_xgb.pickle', 'rb') as f:
        ames_xgb = pickle.load(f)
else:
    # STUDENTS: RUN THE LINE BELOW ONLY:
    ames_xgb = GridSearchCV(estimator=xgb.XGBRegressor(
    objective='reg:linear',
    learning_rate =0.1,
    random_state=42),
    param_grid=param_test,
    n_jobs=4,
    iid=False,
    cv=10)
    ames_xgb.fit(ames_train_X, ames_train_y)
    pickle.dump(ames_xgb, open('models/ames_xgb.pickle', 'wb'))
    


# print(gsearch1.grid_scores_)
print(ames_xgb.best_params_)
print(ames_xgb.best_score_)
```

    [16:26:42] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    {'colsample_bytree': 0.5, 'gamma': 0, 'max_depth': 3, 'min_child_weight': 0.5, 'n_estimators': 285, 'subsample': 0.75}
    0.9138998720966125


Unfortunately, even the code above runs out of RAM on my normal machine, so I optimised it all on Artemis HPC. I then ran code very similar to what we've used before to get the training and testing RMSE.


```python
# What was the RMSE on the training data?
assess_model_fit(models=[ames_ols_all, ames_ridge, ames_lasso, ames_enet, ames_pcr, ames_plsr, ames_mars, ames_RF, ames_knn, ames_gbm, ames_xgb],
                 model_labels=['OLS','Ridge', 'Lasso', 'ENet','PCR','PLSR','MARS', 'RF', 'kNN', 'GB', "XGB"],
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
      <th>GB</th>
      <td>11528.128</td>
      <td>0.979</td>
      <td>7274.574</td>
    </tr>
    <tr>
      <th>XGB</th>
      <td>13901.512</td>
      <td>0.969</td>
      <td>9546.663</td>
    </tr>
    <tr>
      <th>OLS</th>
      <td>15757.714</td>
      <td>0.961</td>
      <td>10931.668</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>16308.709</td>
      <td>0.958</td>
      <td>11079.531</td>
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
      <td>16752.591</td>
      <td>0.955</td>
      <td>11765.495</td>
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
# What was the RMSE on the test data?

assess_model_fit(models=[ames_ols_all, ames_ridge, ames_lasso, ames_enet, ames_pcr, ames_plsr, ames_mars, ames_RF, ames_knn, ames_gbm, ames_xgb],
                 model_labels=['OLS','Ridge', 'Lasso', 'ENet','PCR','PLSR','MARS', 'RF', 'kNN', 'GB', "XGB"],
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
      <th>GB</th>
      <td>19131.898</td>
      <td>0.938</td>
      <td>11399.676</td>
    </tr>
    <tr>
      <th>XGB</th>
      <td>19699.189</td>
      <td>0.934</td>
      <td>12359.483</td>
    </tr>
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
      <td>20988.955</td>
      <td>0.925</td>
      <td>13957.143</td>
    </tr>
    <tr>
      <th>MARS</th>
      <td>23226.020</td>
      <td>0.908</td>
      <td>15355.804</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>27458.802</td>
      <td>0.871</td>
      <td>16960.577</td>
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


