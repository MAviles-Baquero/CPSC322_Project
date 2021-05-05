# Name: Tristan Call
# Assignment: PA5
# Date: 2/28/20
# Description: This file holds various training/testing classes

import mysklearn.myutils as myutils
import numpy as np
import math
from mysklearn.mypytable import MyPyTable

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state)
        # seed your random number generator
        # you can use the math module or use numpy for your generator
        # choose one and consistently use that generator throughout your code
       
    
    if shuffle: 
        myutils.randomize_in_place(X, y)
        # shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself

    assert len(X) == len(y)
    N = len(X)
    # If proportion calculate the ceiling
    if isinstance(test_size, float):
        test_size = math.ceil(N * test_size)

    split_index = N - test_size
    
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:] # TODO: fix this

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    folds = [[] for x in range(n_splits)]
    # Distribute data into folds
    for i in range(len(X)):
        fold = i % n_splits
        folds[fold].append(i)

    # Generate train/test folds
    X_train_folds = []
    X_test_folds = []
    for n in range(n_splits):
        X_test_fold = folds[n]

        # All other folds are test folds
        X_train_fold = []
        for fold in folds:
            if fold != X_test_fold:
                X_train_fold = X_train_fold + fold

        X_train_folds.append(X_train_fold)
        X_test_folds.append(X_test_fold)

    return X_train_folds, X_test_folds 

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Group X and y and the row index into a mypytable
    xy = [X[i] + [y[i]] + [i] for i in range(len(X))]
    column_names = ["a" + str(x) for x in range(len(X[0]))] + ["y"] + ["index"]
    table = MyPyTable(column_names=column_names, data=xy)
    # Do a groupby
    col_names, grouped_data = table.group_by("y")

    folds = [[] for x in range(n_splits)]
    # Distribute data into folds
    i = 0
    for subtable in grouped_data:
        for item in subtable:
            fold = i % n_splits
            i += 1
            # Add all but the y attribute
            folds[fold].append(item[-1])
        
    # Generate train/test folds
    X_train_folds = []
    X_test_folds = []
    for n in range(n_splits):
        X_test_fold = folds[n]

        # All other folds are test folds
        X_train_fold = []
        for fold in folds:
            if fold != X_test_fold:
                X_train_fold = X_train_fold + fold

        X_train_folds.append(X_train_fold)
        X_test_folds.append(X_test_fold)

    return X_train_folds, X_test_folds 

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Set up array as 2D filled with 0s, dimensions = len(labels)
    matrix = [[0 for y in range(len(labels))] for x in range(len(labels))]

    # For each true/pred pair, add 1 in the appropriate spot
    for i in range(len(y_true)):
        
        true_val = y_true[i]
        pred_val = y_pred[i]
        matrix[labels.index(true_val)][labels.index(pred_val)] += 1

    return matrix 