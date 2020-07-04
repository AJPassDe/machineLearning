# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:51:56 2020

@author: AJ
"""

# data leakage is and how to prevent it.
# If you don't know how to prevent it, leakage will come up frequently, and it will ruin your models in subtle 
# and dangerous ways. So, this is one of the most important concepts for practicing data scientists.

# Data leakage (or leakage) happens when your training data contains information about the target, 
# but similar data will not be available when the model is used for prediction. 

# 2 main types of leakage : target leakage and train-test contamination

# 1. Target leakage occurs when your predictors include data that will not be available at the time you make predictions

# To prevent this type of data leakage, any variable updated (or created) after the target value 
# is realized should be excluded.

# 2. Train-Test Contamination
# A different type of leak occurs when you aren't careful to distinguish training data from validation data.
















