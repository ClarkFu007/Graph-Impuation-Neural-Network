import numpy as np

datafile_B1 = "Bus_1.csv"
datafile_FA = "Feeder A.csv"
datafile_FB = "Feeder B.csv"
datafile_FC = "Feeder C.csv"
version = "version2"

from my_utils import get_dataset
DataSet = get_dataset(datafile_B1, datafile_FA,
                      datafile_FB, datafile_FC, version=version)
print("DataSet", DataSet.shape)

"""
   Get the modified dataset of different window sizes and
split the dataset into a training set and a test set.
"""
from my_utils import split_tr_te
time_step = 24
split_value, seed_value = 0.7, 0.2

TrainSet, TestSet = split_tr_te(dataset=DataSet, time_step=time_step,
                                split=split_value, seed=seed_value, version=version)
print("TrainSet", TrainSet.shape)
print("TestSet", TestSet.shape)

np.seterr(divide='ignore', invalid='ignore')
from my_utils import iterSVD_fill
iterSVD_fill(test_set=TestSet, missingness=0.2,
             miss_method='nonblock', block_size=6)
iterSVD_fill(test_set=TestSet, missingness=0.4,
             miss_method='nonblock', block_size=6)
iterSVD_fill(test_set=TestSet, missingness=0.5,
             miss_method='nonblock', block_size=6)
iterSVD_fill(test_set=TestSet, missingness=0.06,
             miss_method='block', block_size=8)
iterSVD_fill(test_set=TestSet, missingness=0.1,
             miss_method='block', block_size=4)
iterSVD_fill(test_set=TestSet, missingness=0.2,
             miss_method='block', block_size=3)

