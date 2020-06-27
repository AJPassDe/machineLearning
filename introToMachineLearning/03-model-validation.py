# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:15:06 2020

micro-course:
https://www.kaggle.com/learn/intro-to-machine-learning

dataSet:
https://www.kaggle.com/dansbecker/melbourne-housing-snapshot

@author: AJ
"""

# =============================================================================
# what is model Validation
# =============================================================================

# You You'll want to evaluate almost every model you ever build. In most (though not all)
# applications, the relevant measure of model quality is predictive accuracy. In other words, 
# will the model's predictions be close to what actually happens.

# There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE). 
# Let's break down this metric starting with the last word, error.

# =============================================================================
# # Recap code from 02-your-first-machine-learning-model.py
# =============================================================================

import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# prediction target: select column we want to predict 
# the predcition target is called y by convention
y = melbourne_data.Price  # (stored in a Series)

# multiple features by providing a list
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# the data is called x by convention
X = melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(X, y)

# =============================================================================
# Model validation - MAE "in-sample"
# =============================================================================

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)

# MAE: Mean Absolute Error
# predicion error for each house is: 
# error=actual−predicted
mean_absolute_error(y, predicted_home_prices)

# Easy way to validate: exclude some training data as a validation 

# =============================================================================
# Model Validation - train_test_split with MAE "out-sample"
# =============================================================================

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions)) # MAE

# Your mean absolute error for the in-sample data was about 1116 dollars.
# Out-of-sample it is more than 271,418 dollars.

# There are many ways to improve this model, such as experimenting to find better 
# features or different model types.

# =============================================================================
# Exercises
# =============================================================================

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Step 1: Split your data (train/test)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Step 2: Specify and Fit the Model

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# step 2.1: Inspect predictions

# print the top few validation predictions
print(val_predictions[:5])
# print the top few actual prices from validation data
print(val_y[:5])

# Step 4: Calculate the Mean Absolute Error in Validation Data

# MAE of predictions
val_mae = mean_absolute_error(val_predictions, val_y) # doesnt matter order











