# Name: Tristan Call
# Date: 5/1/21
# Description: This file contains all the used classifiers

import mysklearn.myutils as myutils
from mysklearn.mypytable import MyPyTable
import math
import copy
import random
import mysklearn.myevaluation as myevaluation
from operator import itemgetter

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        # Sum x
        x_sum = 0
        for x in X_train:
            x_sum += x[0]

        mean_x = x_sum / len(X_train)
        mean_y = sum(y_train) / len(y_train)

        top = []
        bot = []
        # Calculate a list for the top/bottom
        for i in range(len(X_train)):
            top.append((X_train[i][0]- mean_x) * (y_train[i] - mean_y))
            bot.append((X_train[i][0] - mean_x) ** 2)

        # Sum them and divide
        m = sum(top) / sum(bot)
        # Rearrange y = mx + b to get b when line goes through the point
        # (mean_x, mean_y)
        b = mean_y - m * mean_x

        self.slope = m
        self.intercept = b

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for item in X_test:
            y = item[0] * self.slope + self.intercept
            y_predicted.append(y)

        return y_predicted 


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train, categorical_cols=None):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 
        self.categorical_cols = categorical_cols

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test e.g. [[.623], [1],...]
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances) e.g. [[0], [4], [2],...]
        """
        # Make sure these are the right length and that X_train has data first
        assert len(X_test[0]) == len(self.X_train[0])
        # K Can't be greater than the X_train length
        assert self.n_neighbors <= len(self.X_train)

        # Would normalize here but not doing that here...
        unsorted_distances = self._calculate_distances(self.X_train, X_test)

        distances, neighbor_indices = self._sort_test_data(unsorted_distances)   

        return distances, neighbor_indices # TODO: fix this
        
    def _calculate_distance(self, X_point, test_point):
        """Calculates the distance between 1 x point and test point

        Args:
            X_point (list of obj)
            test_point (list of obj)

        Returns:
            The distance between them
        """
        sum = 0
        # Calculate sum of (x - y)^2
        for i in range(len(test_point)):
            if self.categorical_cols != None:
                # If data is categorical
                if self.categorical_cols[i] == True:
                    if test_point[i] == X_point[i]:
                        sum += 0
                    else:
                        sum += 1
                else:
                    # Else is continuous
                    sum += (test_point[i] - X_point[i]) ** 2
            else:
                sum += (test_point[i] - X_point[i]) ** 2
        # Square root it
        distance = math.sqrt(sum)
        return distance

    def _calculate_distances(self, X_data, X_test):
        """This calculates the distances between the X data points and 1 test points

        Args:
            column_data (list of list of obj): list of columns of x data
            X_test (list of list of obj)
        
        Returns:
            distance data in 2D column (list of list of obj)
        """
        distances = []
        # For each test instance
        for test in X_test:
            test_distances = []
            # Calculate its distance from all the test sets
            for x_point in X_data:
                distance = self._calculate_distance(x_point, test)
                test_distances.append(distance)
            distances.append(test_distances)

        return distances

    def _sort_list(self, distances):
        """Given a list of the nearest neighbors, returns a sorted list 
        with indexes

        Args:
            distances (list of obj)

        Returns:
            nearest_distances (list of obj) sorted. Size k
            indexes (list of int). Size k

        Note: Regarding the order of duplicates, this is left to the fate
        of the sort/index algorithm. If the sort function keeps the same relative
        order between the duplicates, then the order from the list is maintained
        in the sorted list. However, it can also not do that, in which case the 
        order can vary significantly.
        """
        # Get a list with the K nearest neighbors
        nearest_distances = copy.deepcopy(distances)
        nearest_distances.sort()
        nearest_distances = nearest_distances[:self.n_neighbors]

        # Find the appropriate indexes
        indexes = []
        duplicate = 0
        pre_val = -1
        for val in nearest_distances:
            # Find the index value in the original list
            if pre_val != val:
                index = distances.index(val)

            # Duplicate handling
            else:
                # If duplicate, get next copy
                index = distances.index(val, indexes[-1] + 1)
            indexes.append(index)
            pre_val = val
        return nearest_distances, indexes

    def _sort_test_data(self, distances):
        """Sorts all the test data to get the k nearest neighbors

        Args:
            distances (list of list of obj): 

        Returns:
            sorted_distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test e.g. [[.623], [1],...]
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances) e.g. [[0], [4], [2],...]
        """
        sorted_distances = []
        neighbor_indices = []
        for xtest in distances:
            nearest_distances, indexes = self._sort_list(xtest)
            sorted_distances.append(nearest_distances)
            neighbor_indices.append(indexes)

        return sorted_distances, neighbor_indices


    def _determine_predicted_value(self, indices):
        """Calculates the predicted value

        Args:
            indices (list of obj)

        Returns:
            the majority classification (obj)
        """
        # Get the y values
        y_vals = []
        for i in indices:
            y_vals.append(self.y_train[i])

        # Get the frequencies of each one
        vals, freqs = myutils.get_item_frequency(y_vals)
        # Determine and keep the max
        highest = max(freqs)
        i = freqs.index(highest)
        predicted_val = vals[i]

        return predicted_val

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, indices = self.kneighbors(X_test)
        y_predicted = []
        for test_indices in indices:
            previcted_val = self._determine_predicted_value(test_indices)
            y_predicted.append(previcted_val)



        return y_predicted 

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors (list of float): The prior probabilities computed for each
            label in the training set. 
            The shape is n_priors
        priors_labels (list of str): The labels for each prior. The shape is n_priors
        priors_orig_labels (list of obj): The labels for each prior in their original form (not str). 
            The shape is n_priors
        posteriors(list of list of list of floats): The posterior probabilities computed for each
            attribute value/label pair in the training set.
            The shape is (n_features, n_values, n_priors)
        posteriors_labels (list of list of str): The labels for each posterior
            The shape is (n_features, n_values)

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.priors_labels = None
        self.priors_orig_labels = None
        self.posteriors = None
        self.posteriors_labels = None

    def compute_priors(self, all_data):
        """Computes the priors
        
        Args:
            all_data(mypytable)

        Return:
            priors, priors_labels, priors_orig_labels
        """
        # Group the data by label
        headers, subtables = all_data.group_by("label")
        total = len(all_data.data)
        priors = []
        priors_labels = []
        priors_orig_labels = []

        for subtable in subtables:
            # For each label calculate the prior and record
            attribute_count = len(subtable)
            prior = attribute_count / total
            label = subtable[0][-1]
            priors.append(prior)
            # Grab the original label for results and a str version
            # for internal consistency
            priors_labels.append(str(label))
            priors_orig_labels.append(label)

        return priors, priors_labels, priors_orig_labels

    def compute_posterior(self, subtable):
        """Computes the posteriors for 1 label

        Args:
            subtable (list of list of obj): All data with 1 specific attribute value
                of form (n_samples, n_features + 1 label)

        Return:

        """
        subtable = MyPyTable(column_names=myutils.generate_table_header(self.X_train), data=subtable)
        headers, prior_subtables = subtable.group_by("label")
        posterior_values = []
        total = len(self.X_train)
        # For each prior 
        for i in range(len(self.priors)):
            # Get appropriate prior index
            try:
                # See if probability of specific label exists
                j = headers.index(self.priors_labels[i])
                # Get probability of 
                prob = len(prior_subtables[j]) / total
            except ValueError:
                prob = 0
            
            posterior_values.append(prob / self.priors[i])
        return posterior_values

    def compute_posteriors(self, all_data):
        """Computes the posteriors

        Args:
            all_data(mypytable)
        """
        self.posteriors = []
        self.posteriors_labels = []
        # For each attribute
        for i in range(len(self.X_train[0])):
            headers, subtables = all_data.group_by("a" + str(i))
            posterior_attributes = []
            self.posteriors_labels.append(headers)

            # For each value in that attribute compute the posteriors
            for subtable in subtables:               
                posterior_attributes.append(self.compute_posterior(subtable))
            self.posteriors.append(posterior_attributes)

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        assert len(X_train) != 0
        self.X_train = X_train
        self.y_train = y_train

        headers = myutils.generate_table_header(self.X_train)
        all_data = [X_train[i] + [y_train[i]] for i in range(len(y_train))]
        all_data = MyPyTable(column_names=headers, data=all_data)

        self.priors, self.priors_labels, self.priors_orig_labels = self.compute_priors(all_data)
        #print(self.priors)
        #print(self.priors_labels)
        self.compute_posteriors(all_data)
        #print(self.posteriors)

        #print(self.posteriors_labels)
        
        # optional bonus handling for continuous values

    def predict_test_case(self, test):
        """Computes the probability of a test case 
        
        Args:
            test (list of obj): Shape is n_features
            
        Returns:
            predicted value
        """

        probabilities = []
        # For every prior
        for i in range(len(self.priors)):
            # Multiply by Ci
            prob = self.priors[i]
            # For every feature
            for j in range(len(self.posteriors)):
                # Find the probability of the value given the prior and multiply
                try:
                    value_i = self.posteriors_labels[j].index(str(test[j]))
                except ValueError:
                    # If the value isn't there, the probability is 0
                    prob = 0
                    break
                else:
                    prob *= self.posteriors[j][value_i][i]
                
            probabilities.append(prob)

        # Find max and return. Ties handled randomly by index position
        max_prob = max(probabilities)
        prior = self.priors_orig_labels[probabilities.index(max_prob)]
        return prior


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                #The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        predicted = [self.predict_test_case(test) for test in X_test]

        return predicted 

class MyZeroRClassifier:
    """Represents a zero R classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        prediction (obj): The prediction of the class dataset

        Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyZeroClassifier
        """
        self.X_train = None 
        self.y_train = None
        self.prediction = None

    def fit(self, X_train, Y_train):
        """Fits a MyZeroClassifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = Y_train

        # Group the data by label
        ytrain_2d = [[y] for y in self.y_train]
        all_data = MyPyTable(column_names="label", data=ytrain_2d)
        headers, subtables = all_data.group_by("label")
        
        # Find the most common label
        max_count = len(subtables[0])
        max_value = subtables[0][0][0]
        for subtable in subtables:
            count = len(subtable)
            if count > max_count:
                max_count = count
                max_value = subtable[0][0]

        self.prediction = max_value

    def predict(self, X_test):
        """Prediction function for MyZeroClassifier

        Args: 
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.prediction for x in X_test]

class MyRandomClassifier:
    """Represents a random classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

        Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyRandomClassifier
        """
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, Y_train):
        """Fits a MyRandomClassifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.X_train = X_train
        self.y_train = Y_train

    def predict(self, X_test):
        """Prediction function for MyRandomClassifier

        Args: 
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in X_test:
            # Generate random labels and store them
            i = random.randint(0, len(self.y_train)-1)
            y_predicted.append(self.y_train[i])
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
        attribute_domains(dict of list of obj): Contains attribute names of form {attribute : ["v1", ...]}

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None

    def generate_attribute_domains(self):
        """Generates the attribute domains for all the data
        
        Returns:
            (dict of list of obj): Contains attribute names of form {attribute : ["v1", ...]}
        """
        attribute_domains = {}
        swapped_xtrain = myutils.swap_rows_cols(self.X_train)
        
        # For each attribute
        for i in range(len(self.header) - 1):
            # Get all possible values:
            items, freq = myutils.get_item_frequency(swapped_xtrain[i])
            # Stick in dictionary
            attribute_domains[self.header[i]] = items

        return attribute_domains

    def compute_entropy(self, instances):
        """Computes the entropy of one split

        Args:
            instances (list of list of obj)
        
        Returns:
            entropy(float)
        """
        # Compute priors
        total = len(instances)
        labels = [x[-1] for x in instances]
        items, freqs = myutils.get_item_frequency(labels)
        priors = [x/total for x in freqs]

        # Compute entropy
        entropy = 0
        for prior in priors:
            entropy -= prior * math.log(prior, 2)
        return entropy

    def compute_attribute_entropy(self, instances, attribute):
        """Computes the entropy for one attribute

        Args:
            instances (list of list of obj)
            attribute (str)
        
        Returns:
            entropy value (float)
        """
        # Split with groupby
        partitions = self.partition_instances(instances, attribute)
        entropy_values = []
        weights = []
        # For each value, compute entropy
        for attribute_value, partition in partitions.items():
            weight = len(partition)
            if weight == 0:
                # If empty has an entropy of 0 automatically
                entropy_values.append(0)
            else:
                entropy_values.append(self.compute_entropy(partition))
            weights.append(weight)


        # Compute weighted average
        entropy = 0
        for i in range(len(entropy_values)):
            entropy += entropy_values[i] * weights[i]
        entropy /= len(instances)
        
        return entropy

    def select_attribute(self, instances, available_attributes):
        """Selects an attribute to split on using entropy

        Args:
            instances (list of list of obj)
            available_attributes (list of str)

        Returns:
            the chosen attribute
        """
        min_attribute = available_attributes[0] # Start entropy
        min_entropy = self.compute_attribute_entropy(instances, available_attributes[0])
        # For each attribute determine the entropy
        for attribute in available_attributes:
            entropy = self.compute_attribute_entropy(instances, attribute)
            # Grab the smallest entropy
            if entropy < min_entropy:
                min_entropy = entropy
                min_attribute = attribute

        return min_attribute

    def all_same_class(self, partition):
        """Determines if all the attributes have the same class

        Args:
            partition (list of list of obj)

        Returns:
            False if not, True if yes
        """
        start_class = partition[0][-1]
        for row in partition:
            if row[-1] != start_class:
                return False
        return True       

    def partition_instances(self, current_instances, split_attribute):
        """Splits by levels in attribute domain 
        
        Args:
            current_instances (list of list of obj)
            split_attribute (str) attribute to split on
            
        Returns:
            partitions {value: [instances with that value]}
        """
        attribute_domain = self.attribute_domains[split_attribute]
        attribute_index = self.header.index(split_attribute)
        partitions = {}
        lists = [[] for i in range(len(attribute_domain))]

        # For each value find the index it belongs to and populate list
        for x in current_instances:
            level = x[attribute_index]
            lists[attribute_domain.index(level)].append(x)

        # Fill out partition
        for i in range(len(attribute_domain)):
            partitions[attribute_domain[i]] = lists[i]

        return partitions

    def majority_vote(self, partition):
        """Determines the majority vote of a partition

        Args:
            partition(list of list of obj)
        
        Returns:
            obj
        """
        # Get into right format
        labels = [x[-1] for x in partition]
        items, freq = myutils.get_item_frequency(labels)
        # Parse return data
        highest = max(freq)
        i = freq.index(highest)
        return items[i]

    def tdidt(self, current_instances, available_attributes):
        """Recursive function to create the tree
        
        Args:
            current_instances (list of list of obj)
            available_attributes (list of obj)
            
        Returns:
            tree
        """
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        #print("splitting on: ", split_attribute)
        available_attributes.remove(split_attribute)
        total_in_a = len(current_instances)

        tree = ["Attribute", split_attribute]

        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        #print("partitions:", partitions)

        # for each partition, repeat unless one of the following occurs (base case)
        for attribute_value, partition in partitions.items():
            values_subtree = ["Value", attribute_value]
            # print("Data:")
            # print(split_attribute)
            # for i in range(3):
            #     values = [partition[j][i] for j in range(len(partition))]
            #     items, freq = myutils.get_item_frequency(values)
            #     print("items: ")
            #     print(items)
            #     print("frequencies: ")
            #     print(freq)
            # print(partition)
            # print(len(partition))
            # print(len(available_attributes))
            #print()
            
            
            # CASE 1: all class labels of the partition are the same => make a leaf node
            if len(partition) > 0 and self.all_same_class(partition):
                #print("Case 1")
                # Append the majority value in form
                # ["Leaf", class label, total in leaf, total in attribute]
                class_label = partition[0][-1]
                values_subtree.append(["Leaf", class_label, len(partition), total_in_a])

            # CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(partition) > 0 and len(available_attributes) == 0:
                #print("Case 2")
                class_label = self.majority_vote(partition)
                values_subtree.append(["Leaf", class_label, len(partition), total_in_a])

            # CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(partition) == 0:
                #print("case 3:")
                # Overwrite current tree to backtrack
                class_label = self.majority_vote(current_instances)
                tree = ["Leaf", class_label, total_in_a, 0]
                break
            else:
                subtree = self.tdidt(partition, available_attributes.copy())

                # Handle a case 3, where the attribute doesn't know the total in the above attribute
                if subtree[0] == "Leaf":
                    subtree[-1] = total_in_a
                values_subtree.append(subtree)

            tree.append(values_subtree)

        return tree

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        # calculate headers ["a0", "a1",...]
        self.header = myutils.generate_table_header(self.X_train)
        # calculate attribute_domains dictionary
        self.attribute_domains = self.generate_attribute_domains()
        # Stitch together x_train, y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = self.header[:-1].copy()

        self.tree = self.tdidt(train, available_attributes)
        #print("tree:", self.tree)
        
    def recursive_predict(self, x, tree):
        """Recursively finds the prediction for x

        Args:
            x (list of obj) the test point

        Returns:
            The predicton
        """
        # Grab test value
        cur_attribute = tree[1]
        i = self.header.index(str(cur_attribute))
        test_value = x[i]

        # Find the value in the tree
        for j in range(len(tree) - 2):
            if test_value == tree[j + 2][1]:
                # Grab that subtree
                subtree = tree[j + 2][2]
                # If it's a leaf node, grab attribute then done
                if subtree[0] == "Leaf":
                    return subtree[1]
                else:
                    # Else recursively find it
                    return self.recursive_predict(x, subtree)
                


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for x in X_test:
            #try:
                y_predicted.append(self.recursive_predict(x, self.tree))
            # except:
            #     y_predicted.append("Failed")

        return y_predicted # TODO: fix this

    def determine_decision_rules(self, tree, attribute_names, class_name):
        """Recursively goes through the tree creating decision tree lines

        Args:
            tree (nested list)
            attribute_names(list of str)
            class_name(str)

        Returns:
            decision rules (list of str)
        """
        # Base case
        if tree[0] == "Leaf":
            return [class_name + " == " + str(tree[1])]
        # Else

        # Grab correct attribute name
        if attribute_names == None:
            attribute_name = str(tree[1])
        else:
            i = self.header.index(tree[1])
            attribute_name = attribute_names[i]
        
        decision_rules = []
        # Go through the subtree values
        for j in range(len(tree) - 2):
            value = tree[j + 2][1]
            prefix_str = "IF " + attribute_name + " == " + str(value) + " "
            subtree = tree[j + 2][2]
            local_rules = self.determine_decision_rules(subtree, attribute_names, class_name)
            if len(local_rules) == 1:
                # If the local rules have a length of 1 it must be a leaf node, so add final phrase
                decision_rules.append(prefix_str + "THEN " + local_rules[0])
                
            else: 
                # Else append the new if to the other rules and add to decision_rules
                local_rules = [prefix_str + "AND " + rule for rule in local_rules]
                decision_rules += local_rules

        return decision_rules

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        decision_rules = self.determine_decision_rules(self.tree, attribute_names, class_name)
        for rule in decision_rules:
            print(rule)

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

class MyRandomForestClassifier:
    """Represents a random forest.

    Attributes:
        N (int): Number of trees to generate
        F (int): Number of subsets to include in each tree
        M (int): Number of best trees to keep
        remainder_xtrain(list of list of obj): Remainder x values
        remainder_ytrain(list of list of obj): Remainder y values
        xtrain(list of list of obj): x train for tree construction
        ytrain(list of list of obj): y train for tree construction
        chosen_trees(list of dict): List of all trees
            Structure of {'tree': MyDecisionTree, 'attributes': [str of included], 
                'accuracy': float}
        

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, N, F, M):
        """Initializer for MyRandomForestClassifier.

        Args:
            N (int): Number of trees to generate
            F (int): Number of subsets to include in each tree
            M (int): Number of best trees to keep            
        """
        self.N = N
        self.F = F
        self.M = M
        self.remainder_xtrain = None
        self.remainder_ytrain = None
        self.xtrain = None
        self.ytrain = None
        self.chosen_trees = None


    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        self.generate_remainder_set(X_train, y_train)
        trees = []

        # Get data into right formats
        train = [self.xtrain[i] + [self.ytrain[i]] for i in range(len(self.xtrain))]
        headers = myutils.generate_table_header(self.xtrain)
        # Set F to valid value
        f = self.F
        if f > len(headers):
            f = len(headers)

        # Generate trees
        for i in range(self.N):
            # Grab random attribute subset and compute bootsrap
            tree = {}
            tree['attributes'] = myutils.compute_random_subset(headers[:-1], f)
            train_set, validation_set = myutils.compute_bootstrap(train)
            # Get right data based on attribute subset
            # Note: Returns as mypytables
            train_set = myutils.get_columns_array(train_set, headers, tree['attributes'] + ['label'])
            validation_set = myutils.get_columns_array(validation_set, headers, tree['attributes'] + ['label'])


            tree['tree'] = self.generate_tree(train_set.data)
            # Determine accuracy
            self.test_tree(tree, validation_set.data)
            trees.append(tree)

        # Find best M of N trees
        self.chosen_trees = self.find_best_trees(trees)
        print(self.chosen_trees)

    def find_best_trees(self, trees):
        """Finds the best N trees

        Args:
            trees (list of dict)
        
        Returns:
            chosen trees
        """
        sorted_list = sorted(trees, key=itemgetter('accuracy'), reverse=True)
        return sorted_list[:self.N - 1]


    def test_tree(self, tree, validation_set):
        """Tests the tree and computes the accuracy

        Args:
            tree (dict)
            validation_set (list of list of obj)
        Note: Includes accuracy in tree dictionary
        """
        xdata = [row[:-1] for row in validation_set]
        actual_values = [row[-1] for row in validation_set]
        predicted_values = tree['tree'].predict(xdata)
        tree['accuracy'] = myutils.calculate_accuracy(predicted_values, actual_values)

    def generate_tree(self, train):
        """This function generates a decision tree

        Args:
            train(list of list of numeric vals): The list of training samples with class labels
                The shape of X_train is (n_train_samples, n_features)
        
        Returns:
            MyDecisionTree
        """
        tree = MyDecisionTreeClassifier()
        # Split data up
        xdata = [row[:-1] for row in train]
        ydata = [row[-1] for row in train]
        tree.fit(xdata, ydata)
        return tree

    def generate_remainder_set(self, X_train, y_train):
        """Generates the remainder set and test set

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        X_train_folds, X_test_folds = myevaluation.stratified_kfold_cross_validation(X_train, y_train, n_splits=3)
        # Grab a random remainder fold
        i = random.randrange(0, 3)
        remainder_fold = X_test_folds[i]
        self.remainder_xtrain = myutils.distribute_data_by_index(X_train, remainder_fold)
        self.remainder_ytrain = myutils.distribute_data_by_index(y_train, remainder_fold)

        # Grab train fold
        train_fold = X_train_folds[i]
        self.xtrain = myutils.distribute_data_by_index(X_train, train_fold)
        self.ytrain = myutils.distribute_data_by_index(y_train, train_fold)


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        pass

    def test_tree_performance(self):
        """Tests the forest's performance against the remainder set

        Returns: 
            (str) with performance data
        """