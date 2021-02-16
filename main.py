import matplotlib.pyplot as plt
import test_and_train_data as tatd
from pandas.plotting import scatter_matrix
import prep_data as prep
import random_forest_reg as rfr

# load data
df = tatd.load_data()
# split data in train and test set
train, test = tatd.split_train_test(df)

# split the label column from the feature columns
housing, housing_labels = prep.split_labels(train)
# impute missing values, add combined attributes and one hot encode
# categorical data
housing_prepared = prep.transform_data(housing)

# train and evaluate decision tree model
scores = rfr.forest_regression(housing_prepared, housing_labels)
rfr.display_scores(scores)

#




