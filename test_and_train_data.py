import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit


def load_data():
    housing = pd.read_csv("C:/Users/carol/PycharmProjects/housing_project/datasets/housing/housing.csv")
    return housing


def visualize_data(housing):
    desired_width = 320
    pd.set_option("display.width", desired_width)
    pd.set_option("display.max_column", 20)
    # housing.info()
    # print(housing["ocean_proximity"].value_counts())
    # print(housing.describe())
    # housing.hist(bins=50, figsize=(20,15))
    # plt.show
    print(housing.head())


def split_train_test(housing):
    # use stratified sampling to define the test dataset
    housing['income_cat'] = pd.cut(housing['median_income'],
                               bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels = [1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    # remove the income_cat attribute so the data is back to its original state
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    return strat_train_set, strat_test_set











