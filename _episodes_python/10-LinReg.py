import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import statsmodels.api as sm
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

# %matplotlib inline
sns.set(font_scale = 1.5)


def onehot_row_drop(df, column = 'column'):
    df = pd.concat([df.drop(column, axis=1), pd.get_dummies(df[column], drop_first=True, prefix=column)],axis=1)
    return(df)

def group_infrequent(df, columnname,counts = 10):
    df.loc[df[columnname].value_counts()[df[columnname]].values <= counts, columnname] = "Other"


def none_of_this_feature(df, column = 'column'):
    # adds a column to the dataframe wtih a value of 0 where there is none of this feature,
    # and a 1 where it's actually there
    df['No_' + column] = np.where(df[column]==0, 0, 1)
    return(df)





ameshousingClean = pd.read_csv('data/AmesHousingClean.csv')

ameshousingClean['Age'] = ameshousingClean['Year_Sold'].max() - ameshousingClean['Year_Built']
ameshousingClean['Remodel_Age'] = ameshousingClean['Year_Sold'].max() - ameshousingClean['Year_Remod_Add']
ameshousingClean['Misc_Feature_Present'] = np.where(ameshousingClean['Misc_Feature'] == "None", 0, 1)

# group situations where there are less than 20 cases of something
for i in ['Neighborhood', 'Roof_Matl', 'Exterior_1st', 'Exterior_2nd', 'Heating', 'MS_Zoning', 'Misc_Feature']:
    group_infrequent(df=ameshousingClean, columnname=i, counts=20)

# Capture variables where some houses have "none of this feature" (i.e value of parameter = 0 aka zero-inflation)
none_of_feature = ['Second_Flr_SF', 'Three_season_porch', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Enclosed_Porch',
                   'Low_Qual_Fin_SF',
                   'Mas_Vnr_Area', 'Lot_Frontage', 'Open_Porch_SF', 'Screen_Porch', 'Pool_Area', 'Wood_Deck_SF',
                   'BsmtFin_SF_2', 'Second_Flr_SF']
for i in none_of_feature:
    ameshousingClean = none_of_this_feature(df=ameshousingClean, column=i)

categorical_variable_list = ['Alley', 'Bldg_Type', 'Condition_1', 'Electrical', 'Exter_Cond', 'Exter_Qual',
                             'Foundation', 'Functional', 'House_Style', 'Kitchen_Qual', 'Land_Contour',
                             'Land_Slope', 'Lot_Config', 'Lot_Shape', 'Central_Air', 'Bsmt_Cond',
                             'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_Type_2', 'Bsmt_Qual', 'Fence',
                             'Fireplace_Qu', 'Garage_Cond', 'Garage_Finish', 'Garage_Type', 'Heating_QC',
                             'MS_SubClass', 'Overall_Cond', 'Overall_Qual',
                             'Paved_Drive', 'Roof_Style', 'Year_Sold', 'Neighborhood', 'Roof_Matl', 'Exterior_2nd',
                             'Exterior_1st', 'Heating', 'MS_Zoning', 'Misc_Feature', 'Street', 'Mas_Vnr_Type',
                             'Utilities', "Condition_2"]

# one hot encode the other categoricals
for i in categorical_variable_list:
    ameshousingClean = onehot_row_drop(df=ameshousingClean, column=i)

# drop double no basement: (ranges from 80 to 83 in dataset, going with 80)
ameshousingClean = ameshousingClean.drop('Bsmt_Exposure_No_Basement', axis=1)
ameshousingClean = ameshousingClean.drop('BsmtFin_Type_2_No_Basement', axis=1)
ameshousingClean = ameshousingClean.drop('Bsmt_Qual_No_Basement', axis=1)
ameshousingClean = ameshousingClean.drop('Garage_Finish_No_Garage', axis=1)
ameshousingClean = ameshousingClean.drop('Garage_Type_No_Garage', axis=1)
ameshousingClean = ameshousingClean.drop('Year_Built', axis=1)
ameshousingClean = ameshousingClean.drop('Year_Remod_Add', axis=1)


ameshousingClean['Sale_Price_quartile'] =  pd.qcut(ameshousingClean['Sale_Price'], 10, labels=range(10))


index_train, index_test  = train_test_split(np.array(ameshousingClean.index), train_size=0.7, test_size = 0.3,
                                            stratify = np.array(ameshousingClean['Sale_Price_quartile']), random_state=42)
ameshousingClean = ameshousingClean.drop('Sale_Price_quartile', axis = 1)

# Create variables for the training and test sets
ames_train = ameshousingClean.loc[index_train,:].copy()
ames_test =  ameshousingClean.loc[index_test,:].copy()


# What are their dimensions?
print(ames_train.shape)
print(ames_test.shape)


predictors = list(ameshousingClean.columns)
predictors.remove('Sale_Price')



# Create training and test response vectors
ames_train_y = np.log(ames_train['Sale_Price'])
ames_test_y = np.log(ames_test['Sale_Price'])

# Write training and test design matrices
ames_train_X = ames_train[predictors].copy()
ames_test_X = ames_test[predictors].copy()

ols = LinearRegression()
ols.fit(ames_train_X, ames_train_y)


ameshousingClean.head()