import random
import itertools
import numpy as np
import scipy.stats as stats 
import mysklearn.myutils as myutils

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

# pasted from DecisionTreeFun
header = ["level", "lang", "tweets", "phd"]
attribute_domains = {"level": ["Senior", "Mid", "Junior"], 
    "lang": ["R", "Python", "Java"],
    "tweets": ["yes", "no"], 
    "phd": ["yes", "no"]}
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

def test_forest_fit():
    random.seed(0)
    forest = MyRandomForestClassifier(3, 2, 2)
    forest.fit(X, y)



    assert False == True
def test_forest_predict():
    assert False == True