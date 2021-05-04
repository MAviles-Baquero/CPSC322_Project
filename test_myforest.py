import random
import itertools
import numpy as np
import scipy.stats as stats 
import mysklearn.myutils as myutils

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForestClassifier

# posted from DecisionTreeFun
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
xtest = [["Mid", "R", "no", "no"]]
xtest_solution = ["True"]
xtest2 = [["Junior", "R", "yes", "no"], ["Senior", "Python", "yes", "yes"]]
xtest2_solution = ["True", "False"]


"""
a0 -> level
a1 -> lang
a2 -> tweets
a3 -> phd
"""

# attributes of a2 and a3 (Tweets and PhD)
first_decision_tree = [
    ['Attribute', 'a2', 
        ['Value', 'Mid', 
            ['Leaf', 'True', 3, 14]
        ], 
        ['Value', 'Senior', 
            ['Attribute', 'a3', 
                ['Value', 'R', 
                    ['Leaf', 'True', 2, 6]
                ], 
                ['Value', 'Python', 
                    ['Leaf', 'False', 3, 6]
                ], 
                ['Value', 'Java', 
                    ['Leaf', 'False', 1, 6]
                ]
            ]
        ], 
        ['Value', 'Junior', 
            ['Leaf', 'True', 5, 14]
        ]
    ]
]


# attributes of a2 and a0 (Tweets and Level)
second_decision_tree = [
    ['Attribute', 'a2', 
        ['Value', 'Mid', 
            ['Leaf', 'True', 3, 14]
        ], 
        ['Value', 'Senior', 
            ['Attribute', 'a0', 
                ['Value', 'Python', 
                    ['Leaf', 'False', 2, 7]
                ], 
                ['Value', 'R', 
                    ['Leaf', 'True', 1, 7]
                ], 
                ['Value', 'Java', 
                    ['Leaf', 'False', 4, 7]
                ]
            ]
        ], 
        ['Value', 'Junior', 
            ['Leaf', 'True', 4, 14]
        ]
    ]
]


def test_forest_fit():
    # We confirmed that the classifier gives random results by repeatedly running it
    # Unfortunately multiple instances of random are called, making it very difficult to 
    # test those. Especially since the decision tree itself also uses random...
    # As such we just grabbed the random attribute results and tested the resulting
    # trees here, instead of trying to predict those values otherwise
    random.seed(1)
    forest = MyRandomForestClassifier(3, 2, 2)
    forest.fit(X, y)
    
    trees = forest.chosen_trees
    tree = trees[0]
    
#     for i in range(len(trees)):
#         print(sort[i].chosen_trees[0])
        
#     print(tree['tree'])
        
    assert tree['attributes'] == ['a2', 'a3']
    assert tree['tree'] == first_decision_tree

    tree = trees[1]
    assert tree['attributes'] == ['a2', 'a0']
    assert tree['tree'] == second_decision_tree

def test_forest_predict():
    random.seed(1)
    forest = MyRandomForestClassifier(3, 2, 2)
    forest.fit(X, y)
    assert forest.predict(xtest) == xtest_solution
    assert forest.predict(xtest2) == xtest2_solution

    assert False == True