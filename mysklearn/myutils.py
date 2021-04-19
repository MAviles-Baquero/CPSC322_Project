# Name: Tristan Call
# Assignment: PA4
# Date: 2/28/20
# Description: This file holds generic functions

from mysklearn.mypytable import MyPyTable 
import mysklearn.plot_utils
import numpy as np 
import math

def compute_equal_width_cutoffs(values, num_bins):
    """ This computes equal width cutoffs for a given number of bins

    Args:
        values (list of float obj)
        num_bins (int)
    """
    values_range = max(values) - min(values)
    size = values_range / num_bins
    # N + 1 cutoffs
    # bin width probably floating point # so handle with np
    cutoffs = list(np.arange(min(values), max(values), size))
    cutoffs.append(max(values))
    cutoffs = [round(x, 2) for x in cutoffs]
    return cutoffs

def compute_frequency_labels(cutoffs):
    """Compute frequency labels for a given set of cutoffs

    Args:
        cutoffs (list of obj)
    
    Returns:
        labels (list of str)
    """
    labels = []
    for i in range(len(cutoffs)-1):
        # Create the label
        label = "[" + str(cutoffs[i]) + ", " + str(cutoffs[i+1])
        if i < len(cutoffs) - 2:
            label += ")"
        else:
            label += "]"
        # Add to labels
        labels.append(label)
    return labels

def compute_frequencies(values, cutoffs):
    """Calculates the frequencies of each values based on the cutoffs

    Args:
        values (list of floats)
        cutoffs (list of floats): bin cutoffs

    Returns:
        frequencies (list of ints)
    """
    frequencies = []
    # For each cutoff section
    for i in range(len(cutoffs) - 1):
        lower_val = cutoffs[i]
        upper_val = cutoffs[i + 1]
        frequency = 0
        # If in [lower_val, upper_val) range add
        for val in values:
            if val >= lower_val and val < upper_val:
                frequency += 1

        # Add on the max values at the end for [lower_val, upper_val]
        if i == len(cutoffs) - 2:
            for val in values:
                if val == upper_val:
                    frequency += 1

        frequencies.append(frequency)

    return frequencies

def generate_slope_intercept(x, y):
    """Generates the m and b of a y = mx + b linear regression line

    Args:
        x (list of float): x values
        y (list of float): y values
    
    Returns
        m and b
    """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    top = []
    bot = []
    # Calculate a list for the top/bottom
    for i in range(len(x)):
        top.append((x[i]- mean_x) * (y[i] - mean_y))
        bot.append((x[i] - mean_x) ** 2)

    # Sum them and divide
    m = sum(top) / sum(bot)
    # Rearrange y = mx + b to get b when line goes through the point
    # (mean_x, mean_y)
    b = mean_y - m * mean_x

    return m, b

def generate_std(x):
    """This function computes the standard deviation

    Args:
        x (list of float): x values

    Returns:
        std
    """

    mean = sum(x) / len(x)
    smd = [((xi - mean) ** 2) for xi in x]
    variance = sum(smd) / len(smd)
    std = math.sqrt(variance)
    return std

def generate_correlation_r_covar(x, y):
    """Generates the r and covariance of a correlation

    Args:
        x (list of float): x values
        y (list of float): y values
    
    Returns
        r
    """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    top = []
    x_sum = []
    y_sum = []
    # Calculate a list for the covariance/bottom
    for i in range(len(x)):
        top.append((x[i]- mean_x) * (y[i] - mean_y))
        x_sum.append((x[i]- mean_x) ** 2)
        y_sum.append((y[i]- mean_y) ** 2)

    # Sum them and divide
    covariance = sum(top) / len(top)
    x_std = generate_std(x)
    y_std = generate_std(y)


    r = covariance / x_std / y_std


    return r, covariance

def convert_list_to_double_list(xlist):
    """Convert a list of the form [1,2,3] to the form [[1], [2], [3]

    Args:
        list (list of obj)

    Returns:
        list of list of obj
    """
    new_list = []
    for x in xlist:
        new_list.append([x])
    return new_list

def swap_rows_cols(array):
    """This swaps the array rows and columns

    Args:
        array (list of list of obj)

    Returns:
        list of normalized columns (list of list of obj)
    """
    # Create table with disposable headers
    header = ["a" + str(i) for i in range(len(array[0]))]
    table = MyPyTable(header, array)

    # Get each column and normalize it
    column_data = []
    for i in range(0, len(array[0])):
        col = table.get_column(i)
        column_data.append(col)

    return column_data

def get_item_frequency(xlist):
    """Returns the frequencies of items in a list

    Args:
        xlist (list of obj)

    Returns:
        items (list of obj)
        freq (list of ints)
    """
    items = []
    freq = []
    for x in xlist:
        try:
            # iterate freq
            i = items.index(x)
            freq[i] += 1
        except ValueError:
            # Add new item
            items.append(x)
            freq.append(1)

    return items, freq

def randomize_in_place(alist, parallel_list=None):
    """Swaps around the contents of a list, and optionally a parallel list too

    Args:
        alist(list of obj)
        parallel_list(list of obj or None)
    """ 
    for i in range(len(alist)):
        rand_index = np.random.randint(0, len(alist))
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def categorize_continuous(value, cutoffs, categories):
    """categorizes a continuous value based on cutoffs

    Args:
        value (float)
        cutoffs (list of floats): bin cutoffs
        categories (list of obj)

    Return:
        Appropriate category (obj)
    """
    for i in range(len(cutoffs) - 1):
        lower_val = cutoffs[i]
        upper_val = cutoffs[i + 1]
        # If in [lower_val, upper_val) range replace value
        # or if last value in cutoffs
        if value >= lower_val and value < upper_val:
                return categories[i]

        # Add on the max values at the end for [lower_val, upper_val]
        if i == len(cutoffs) - 2:
            if value == upper_val:
                return categories[i]

def categorize_continuous_list(list_vals, cutoffs, categories):
    """categorizes a continuous value based on cutoffs

    Args:
        list_vals (list of float)
        cutoffs (list of floats): bin cutoffs
        categories (list of obj)

    Return:
        Appropriate category (list of obj)
    """
    new_list = []
    for i in range(len(list_vals)):
        new_list.append(categorize_continuous(list_vals[i], cutoffs, categories))
    return new_list

def normalize_cols(X, X_test):
    """Normalizes all columns and the corresponding test data

    Args:
        column_data (list of list of obj): X_train data
        X_test (list of list of obj): X_test data

    Returns:
        column_data (list of list of obj): X_train data normalized  
        X_test (list of list of obj): X_test data normalized

    NOTE: For continuous values only!
    """
    # Swap into column form to simplify normalizing
    column_data = swap_rows_cols(X)
    test_col_data = swap_rows_cols(X_test)
    # Normalize every attribute
    for i in range(len(column_data)):
        normalize(column_data[i], test_col_data[i])

    # Swap back and return
    column_data = swap_rows_cols(column_data)
    test_col_data = swap_rows_cols(test_col_data)
    return column_data, test_col_data


def normalize(col, test_col):
    """This normalizes a column and test column based on its min and max value

    Args:
        col (list of obj)
        test_col (list of obj)

    NOTE: For continuous values only!
    """
    col_min = min(col)
    normalizer = max(col) - col_min
    # Normalize the column
    for i in range(len(col)):
        col[i] = (col[i] - col_min) / normalizer

    # Normalize test column
    for j in range(len(test_col)):
        test_col[j] = (test_col[j] - col_min) / normalizer
        # Handle cases of test col outside of normal range
        if test_col[j] > 1:
            test_col[j] = 1
        elif test_col[j] < 0:
            test_col[j] = 0

def calculate_accuracy(predicted, actual):
    """Calculates the accuracy

    Args:
        predicted (list of obj)
        actual (list of obj)
    
    Returns:
        Accuracy (float)
    """
    right_num = 0
    assert len(predicted) == len(actual)
    for i in range(len(predicted)):
        # See if each accurate
        if predicted[i] == actual[i]:
            right_num += 1
    # Calculate accuracy
    accuracy = right_num / len(predicted)
    return accuracy

def distribute_data_by_index(data, indices):
    """Creates a list of the data at the indices in indices

    Args:
        data (list of obj)
        indices (list of int)
    
    Returns:
        data_subset (list of obj)
    """
    data_subset = []
    for i in range(len(indices)):
        data_subset.append(data[indices[i]])
    return data_subset

def format_confusion_matrix_into_table(matrix, labels, classifier_name):
    """Formats the confusion matrix into a pretty table

    Args:
        matrix (list of list of int)
        labels (list of obj)
        classifier_name (string)

    Returns:
        header (list of str)
    """
    for i in range(len(matrix)):
        # Calculate total/% accuracy
        total = sum(matrix[i])
        # The [i][i] value are the accurate values
        if total != 0:
            recognition = matrix[i][i] / total
        else:
            recognition = 0

        # Attach label, total, and recognition to row
        matrix[i].append(total)
        matrix[i].append(round(recognition * 100, 2))
        matrix[i].insert(0, labels[i])

    # Format the header
    header = [classifier_name] + labels + ["Total"] + ["Recognition (%)"]
    return header

def split_col_off(table, i):
    """Splits a column off from the table

    Args:
        table (list of list of obj)
        i (int): index of col to remove

    Return:
        table without the column
        the column
    """
    new_table = []
    col = []
    for row in table:
        col.append(row[i])
        new_table.append(row[:i] + row[i+ 1:])
    return new_table, col

def compare_3d_list(lista, listb):
    """Compares 2 lists, assert all layers are equal
    
    Args:
        lista (list of list of list of obj)
        listb (list of list of list of obj)
    """
    rtol = 1e-03
    for i in range(len(lista)):
        assert np.allclose(lista[i], listb[i], rtol)

def generate_table_header(X_train):
    """Generates a generic header for tables with the standard number of
    attributes and 1 label column

    Args: 
        X_train(list of list of obj): The list of training instances (samples). 
            The shape of X_train is (n_train_samples, n_features)

    Returns:
        headers (list of str)
    """
    headers = ["a" + str(i) for i in range(len(X_train[0]))] + ["label"]
    return headers

