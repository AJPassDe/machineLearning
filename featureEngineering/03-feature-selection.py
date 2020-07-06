# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 12:58:45 2020

@author: AJ
"""

# Often you'll have hundreds or thousands of features after various encodings and feature generation. 
# This can lead to two problems. First, the more features you have, the more likely you are to overfit to 
# the training and validation sets. This will cause your model to perform worse at generalizing to new data.

# Secondly, the more features you have, the longer it will take to train your model and optimize hyperparameters. 
# Also, when building user-facing products, you'll want to make inference as fast as possible. Using fewer features
# can speed up inference at the cost of predictive performance.

# To help with these issues, you'll want to use feature selection techniques to keep the most informative features 
# for your model.

# =============================================================================
# 2) Univariate Feature Selection
# =============================================================================

from sklearn.feature_selection import SelectKBest, f_classif
feature_cols = clicks.columns.drop(['click_time', 'attributed_time', 'is_attributed'])
train, valid, test = get_data_splits(clicks)

# Create the selector, keeping 40 features
selector = SelectKBest(f_classif, k=40)

# Use the selector to retrieve the best features
X_new = selector.fit_transform(train[feature_cols], train['is_attributed']) 

# Get back the kept features as a DataFrame with dropped columns as all 0s
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=train.index, 
                                 columns=feature_cols)

# Find the columns that were dropped
dropped_columns = selected_features.columns[selected_features.var() == 0]


# =============================================================================
# 4) Use L1 regularization for feature selection
# =============================================================================


def select_features_l1(X, y):
    logistic = LogisticRegression(C=0.1, penalty="l1", random_state=7, solver='liblinear').fit(X, y)
    model = SelectFromModel(logistic, prefit=True)

    X_new = model.transform(X)

    # Get back the kept features as a DataFrame with dropped columns as all 0s
    selected_features = pd.DataFrame(model.inverse_transform(X_new),
                                    index=X.index,
                                    columns=X.columns)

    # Dropped columns have values of all 0s, keep other columns
    cols_to_keep = selected_features.columns[selected_features.var() != 0]