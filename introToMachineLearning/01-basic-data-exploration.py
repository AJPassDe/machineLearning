# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:48:56 2020

micro-course:
https://www.kaggle.com/learn/intro-to-machine-learning

dataSet:
https://www.kaggle.com/dansbecker/melbourne-housing-snapshot

@author: AJ
"""


# 1 step in ML --> familiarize with the data --> use pandas 
# pandas: explore and manipulate data

# =============================================================================
# Pandas
# =============================================================================

import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# print a summary of the data in Melbourne data
melbourne_data.describe()

# count: shows how many rows ghave non-missing values


# step 1: loading data 
 
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# print a summary of the data in iowa data
summaryIowa = home_data.describe()

