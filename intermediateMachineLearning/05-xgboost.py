# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:28:17 2020

@author: AJ
"""
# Ensemble methods combiene the predictions of several models (e.g. several trees, in the case of random forest)

# Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# n_estimators specifies how many times to go through the modeling cycle described above. 
# It is equal to the number of models that we include in the ensemble.

# Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
# Too high a value causes overfitting, which causes accurate predictions on training data,
# but inaccurate predictions on test data (which is what we care about).

# Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))



# early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. 
# Early stopping causes the model to stop iterating when the validation score stops improving, 
# even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators 
# and then use early_stopping_rounds to find the optimal time to stop iterating.

# Since random chance sometimes causes a single round where validation scores don't improve, 
# you need to specify a number for how many rounds of straight deterioration to allow before stopping. 
# Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds 
# of deteriorating validation scores.

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, 
# though it will also take the model longer to train since it does more iterations through the cycle. 
# As default, XGBoost sets learning_rate=0.1.

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# n_jobs
# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster.
# It's common to set the parameter n_jobs equal to the number of cores on your machine.
# On smaller datasets, this won't help.

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# XGBoost is a the leading software library for working with standard tabular data 
# (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). 
# With careful parameter tuning, you can train highly accurate models.
















