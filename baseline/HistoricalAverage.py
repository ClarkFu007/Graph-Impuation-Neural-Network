import numpy as np
from my_utils import get_dataset
from my_utils import split_tr_te
from my_utils import historical_average


def main():
    # Input the original dataset.
    datafile_B1 = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Bus_1.csv"
    datafile_FA = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Feeder A.csv"
    datafile_FB = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Feeder B.csv"
    datafile_FC = r"C:/My Files/Academic Files/Graduate Study/" \
                  "University of Southern California/Academic Material/" \
                  "Research/Codes/ginn/Smart Meter Data/Dataset/Baseline/Feeder C.csv"
    dataSet = get_dataset(datafile_B1, datafile_FA, datafile_FB, datafile_FC, version="version1")
    print("The original dataSet", dataSet.shape)

    # Get the training set, validation set, and test set
    time_step = 24
    train_num, validation_num = 275, 30
    trainSet, validationSet, testSet = \
        split_tr_te(dataset=dataSet, time_step=time_step, train_num=train_num,
                    validation_num=validation_num, version="version1")
    print("Training set", trainSet.shape)
    print("Validation set", validationSet.shape)
    print("Test set", testSet.shape)
    print(" ")

    historical_average(test_set=testSet, missingness=0.2,
                       miss_method='nonblock', block_size=1)
    historical_average(test_set=testSet, missingness=0.4,
                       miss_method='nonblock', block_size=1)
    historical_average(test_set=testSet, missingness=0.5,
                       miss_method='nonblock', block_size=1)
    historical_average(test_set=testSet, missingness=0.06,
                       miss_method='block', block_size=8)
    historical_average(test_set=testSet, missingness=0.08,
                       miss_method='block', block_size=8)
    historical_average(test_set=testSet, missingness=0.1,
                       miss_method='block', block_size=4)
    historical_average(test_set=testSet, missingness=0.15,
                       miss_method='block', block_size=4)
    historical_average(test_set=testSet, missingness=0.18,
                       miss_method='block', block_size=3)
    historical_average(test_set=testSet, missingness=0.2,
                       miss_method='block', block_size=3)


if __name__ == '__main__':
    main()
