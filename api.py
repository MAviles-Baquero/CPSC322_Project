import json
import datetime
from flask import Flask, jsonify, request
import os
import pickle
import mysklearn.myclassifiers
from mysklearn.myclassifiers import MyRandomForestClassifier, MyDecisionTreeClassifier
import mysklearn.myutils as myutils


app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome to my app", 200

@app.route("/predict")
def predict():
    """This allows the user to input data and get a prediction"""
    birth_year = int(request.args.get("birth_year", ""))
    gender = int(request.args.get("gender", ""))
    hispanic = int(request.args.get("hispanic", ""))
    race = int(request.args.get("race", ""))
    income = int(request.args.get("income", ""))
    education = int(request.args.get("education", ""))

    # Error checking
    if (not (gender >= 1 and gender <= 2)
        or not (hispanic >= 1 and hispanic <= 2)
        or not (race >= 1 and race <= 4)
        or not (income >= 1 and income <= 8)
        or not (education >= 1 and education <= 7)
        or not (birth_year >= 1932 and birth_year <= 2002)):
        return "Error in string parsing", 400

    # classify birth_year
    year_label = [x + 1 for x in range(7)]
    cutoffs = [1932 + 10 * x for x in range(8)]
    birth_year = myutils.categorize_continuous_list([birth_year], cutoffs, year_label)

    # Test the classifier
    xtest = [[birth_year[0], gender, hispanic, race, income, education]]
    print(xtest)
    forest = get_forest()
    print(forest.chosen_trees)
    result = forest.predict(xtest)    
    print(result)

    if result[0] == 1:
        result = "Delayed or canceled"
    else:
        result = "not delayed or canceled"

    result = {"Delayed/canceled status": result}
    return jsonify(result), 200

def get_forest():
    """This function unpacks the pickle and gets all the variables in appropriate places

    Returns:
        forest
    """
    infile = open("forest_pickler.py", "rb")
    forest = MyRandomForestClassifier(1, 2, 2)
    trees = pickle.load(infile)
    chosen_trees = []
    # Recompile each tree
    for tree in trees:
        dict = {}
        decisionTree = MyDecisionTreeClassifier()
        decisionTree.tree = tree[0]
        decisionTree.header = tree[1]
        dict['tree'] = decisionTree
        chosen_trees.append(dict)
    forest.chosen_trees = chosen_trees


    infile.close()
    return forest
    


 
if __name__ == "__main__":
    port = of.environ.get("PORT", 5000)
    app.run(debug=FALSE, host = "0.0.0.0", port=port)
    # app.run(debug=True)