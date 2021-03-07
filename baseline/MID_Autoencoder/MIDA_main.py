import time
import numpy as np
from MIDA_utils import get_dataset
from MIDA_utils import split_tr_val_te
from MIDA_core import do_MIDA


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
    dataSet = get_dataset(datafile_B1, datafile_FA, datafile_FB, datafile_FC)
    print("The original dataSet", dataSet.shape)

    # Get the training set, validation set, and test set
    time_step = 24
    train_num, validation_num = 275, 30
    trainSet, validationSet, testSet = \
        split_tr_val_te(dataset=dataSet, time_step=time_step,
                        train_num=train_num,
                        validation_num=validation_num)
    print("Training set", trainSet.shape)
    print("Validation set", validationSet.shape)
    print("Test set", testSet.shape)
    print(" ")


    missingness, corrupt_method = 0.06, 'block'
    block_size = 8
    verbose = False
    """
    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.2, verbose=verbose,
            corrupt_method=corrupt_method, block_size=1, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.4, verbose=verbose,
            corrupt_method=corrupt_method, block_size=1, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.5, verbose=verbose,
            corrupt_method=corrupt_method, block_size=1, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.06, verbose=verbose,
            corrupt_method=corrupt_method, block_size=8, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.08, verbose=verbose,
            corrupt_method=corrupt_method, block_size=8, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.1, verbose=verbose,
            corrupt_method=corrupt_method, block_size=4, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.15, verbose=verbose,
            corrupt_method=corrupt_method, block_size=4, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.18, verbose=verbose,
            corrupt_method=corrupt_method, block_size=3, iter_num=5, theta=8, epochs=100)

    do_MIDA(train_set=trainSet, validation_set=validationSet,
            test_set=trainSet, missingness=0.2, verbose=verbose,
            corrupt_method=corrupt_method, block_size=3, iter_num=5, theta=8, epochs=100)
    """


if __name__ == '__main__':
    main()


