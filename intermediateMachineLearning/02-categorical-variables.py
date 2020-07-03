# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:02:54 2020

@author: AJ
"""

# A categorical variable takes only a limited number of values.

# Consider a survey that asks how often you eat breakfast and provides four options: 
#   "Never", "Rarely", "Most days", or "Every day". 
# In this case, the data is categorical, because responses fall into a fixed set of categories.

# 3 approaches

# 1. Drop categorical variables
# The easiest approach to dealing with categorical variables is to simply remove them from the dataset.
# This approach will only work well if the columns did not contain useful information.

# 2. Label encoding
# Label encoding assigns each unique value to a different integer.
# This approach assumes an ordering of the categories: 
#   "Never" (0) < "Rarely" (1) < "Most days" (2) < "Every day" (3).

# This assumption makes sense in this example, because there is an indisputable ranking to the categories. 
# Not all categorical variables have a clear ordering in the values, but we refer to those that do as ordinal variables. 
# For tree-based models (like decision trees and random forests), you can expect label encoding to work well with ordinal variables.

# 3. One-Hot encoding
# Creates new columns indicating the presence (or absence) of each possible value in the original data. 

# The corresponding one-hot encoding contains one column for each possible value, and one row for each 
# row in the original dataset. 

# This approach to work particularly well if there is no clear ordering in the categorical data 
# (e.g., "Red" is neither more nor less than "Yellow"). 
# We refer to categorical variables without an intrinsic ranking as nominal variables.


import pandas as pd
 from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id') 
X.shape

X_test = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# =============================================================================
# Step 1: Drop columns with categorical data
# =============================================================================

# Fill in the lines below: drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])
print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
# MAE: 17837.82570776256

# =============================================================================
# Step 2: Label encoding
# =============================================================================

print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)


from sklearn.preprocessing import LabelEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder 
label_encoder = LabelEncoder()

for col in good_label_cols:
    label_X_train[col]  = label_encoder.fit_transform(X_train[col])
    label_X_valid[col]  = label_encoder.transform(X_valid[col])
    

print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
# MAE: 17575.291883561644

# =============================================================================
# Step 3: Investigating cardinality
# =============================================================================

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

# MAE: 17525.345719178084






