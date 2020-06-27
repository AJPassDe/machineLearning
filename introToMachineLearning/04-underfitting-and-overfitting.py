# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:13:58 2020

micro-course:
https://www.kaggle.com/learn/intro-to-machine-learning

melbourne dataSet:
https://www.kaggle.com/dansbecker/melbourne-housing-snapshot

iowa dataSet:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

@author: AJ
"""

# This is a phenomenon called overfitting, where a model matches the training 
# data almost perfectly, but does poorly in validation and other new data.

# When a model fails to capture important distinctions and patterns in the data, 
# so it performs poorly even in training data, that is called underfitting.

# There are a few alternatives for controlling the tree depth, and many allow for some routes through the tree to 
# have greater depth than other routes. But the max_leaf_nodes argument provides a very sensible way to control 
# overfitting vs underfitting. The more leaves we allow the model to make, the more we move from the underfitting 
# area in the above graph to the overfitting area.

# We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:

# from sklearn.metrics import mean_absolute_error
# from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# The data is loaded into train_X, val_X, train_y and val_y using the code you've already seen 
# (and which you've already written).

# We can use a for-loop to compare the accuracy of models built with different values for max_leaf_nodes.

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# =============================================================================
# Conclusion
# =============================================================================
 
# Here's the takeaway: Models can suffer from either:

# - Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
# - Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.

# We use validation data, which isn't used in model training, to measure a candidate model's accuracy. 
# This lets us try many candidate models and keep the best one.
    
# =============================================================================
# Recap
# =============================================================================

# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))


# =============================================================================
# Exercises
# =============================================================================

# function to configure get mae with max leaf nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    
    return(mae)

# Step 1: Compare Different Tree Sizes

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for leafs in candidate_max_leaf_nodes:
    maee = get_mae(leafs, train_X, val_X, train_y, val_y)
    print('Num of leafs: %d \t\t mae: %d'%(leafs, maee))
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = 100

#solution answer

# dictionary with leaf nodes: MAE
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
print(scores)
best_tree_size2 = min(scores, key=scores.get) # get number of leaf nodes that makes min MAE
print(best_tree_size2)

# Step 2: Fit Model Using All Data

# Fit the model with best_tree_size. Fill in argument to make optimal size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=0)

# fit the final model
final_model.fit(X, y)



















































