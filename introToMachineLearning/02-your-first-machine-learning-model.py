# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:11:35 2020


micro-course:
https://www.kaggle.com/learn/intro-to-machine-learning

dataSet:
https://www.kaggle.com/dansbecker/melbourne-housing-snapshot

@author: AJ
"""

# =============================================================================
# Preparing data
# =============================================================================

import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# print columns in melbourne_data
melbourne_data.columns

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# =============================================================================
# Selecting precidtion target
# =============================================================================

# prediction target: select column we want to predict 
# the predcition target is called y by convention
y = melbourne_data.Price  # (stored in a Series)

# =============================================================================
# Choosing  "features"
# =============================================================================

# columns inputted into our model ( used later to make predictions) are called "features"
# In our case, those would be the columns used to determine the home price.

# multiple features by providing a list
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# the data is called x by convention
X = melbourne_data[melbourne_features]

# print summary of data in X
X.describe()

# print top few rooms
X.head()

# =============================================================================
# Building your model
# =============================================================================

# steps to build a model:
# 1. define: type of model (e. decision tree), parameters of the model type
# 2. fit: Capture patterns from provided data. Heart of modeling.
# 3. predict: predictiong target
# 4. Evaluate: Determine how accurate the model's predictions are.

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# =============================================================================
# Exercises
# =============================================================================

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# read data and store in df
home_data = pd.read_csv(iowa_file_path)

# column names
home_data.columns

# Step 1: Specify Prediction Target

# prediction target
y = home_data.SalePrice

# Step 2: Create X

# Select features with a list
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
# select data by features
X = home_data[feature_names]

# step 2.1: Review data

# print description or statistics from X
print(home_data.describe())
# print the top few lines
print(home_data.head())

# Step 3: Specify and Fit Model

#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit the model
iowa_model.fit(X, y)

# Step 4: Make Predictions

# make predicitions with model's predict command using X as the data.
predictions = iowa_model.predict(X)
print(predictions)

predictions.head()
X.head()







