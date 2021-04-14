# Name: Tristan Call
# Assignment: PA3
# Date: 2/18/20
# Description: This file holds generic plotting functions

import matplotlib.pyplot as plt


def plot_bar(x, y, xlabel, ylabel, title, *, my_rotation=90, ):
    """Prints a basic bar chart
    
    Args:
        x (list of obj)
        y (list of obj)
        xlabel (string)
        ylabel (string)
        title (string)
        my_rotation (int)
    """
    plt.figure()
    plt.bar(x, y)
    plt.xticks(rotation=my_rotation)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plot_hist(x, xlabel, ylabel, title):
    """Prints a basic histogram

    Args:
        x (list of obj)
        xlabel (string)
        ylabel (string)
        title (string)

    """
    plt.figure()
    plt.hist(x, bins=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def plot_pie_percent(values, labels, title1):
    """Prints a basic pie chart with percentages

    Args:
        values (list of obj)
        labels (list of obj)
        title1 (string)
    """
    plt.figure()
    plt.title(title1)
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    
    plt.show()

def plot_scatter(x, y, xlabel, ylabel, title):
    """Prints a basic scatter plot

    Args:
        x (list of obj)
        xlabel (string)
        ylabel (string)
        title (string)

    """
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(x, y)
    
    
    
