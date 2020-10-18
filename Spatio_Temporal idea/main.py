import numpy as np

datafile_B1 = "Bus_1.csv"
datafile_FA = "Feeder A.csv"
datafile_FB = "Feeder B.csv"
datafile_FC = "Feeder C.csv"
from my_utils import get_dataset
DataSet = get_dataset(datafile_B1, datafile_FA, datafile_FB, datafile_FC)
print("DataSet", DataSet.shape)

"""
   Get the modified dataset of different window sizes and
split the dataset into a training set and a test set.
"""

from my_utils import split_tr_te
time_step = 24
split_value, seed_value = 0.7, 0.2

TrainSet, TestSet = split_tr_te(dataset=DataSet, time_step=time_step,
                                split=split_value, seed=seed_value)
print("TrainSet", TrainSet.shape)
print("TestSet", TestSet.shape)

"""
   Do the imputation.
"""
from my_GINN import do_GINN
L = 4
kernel_size, stride_step = 8, 1
theta = 1

do_GINN(train_set=TrainSet, test_set=TestSet, time_step=time_step,
        missingness=0.1, miss_value=0, miss_method="block", block_size=4,
        theta=theta, epoch_num=0, size_step=16, auto_lr=0.001, fine_tune=False)



