# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb


class Node:
    # tree nodes to make the traversal of the tree easier when predicting the data from the data set
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # storing the nodes
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # helper function to determine a leaf node in the tree
    def is_leaf(self):
        return self.value is not None
    

# this global method will be used to handle the categorical values
def categorical(y):
    # storing it to numpy array
    cat = y.to_numpy()

    # Find the unique elements of an array. Returns the sorted unique elements of an array.
    values, counts = np.unique(cat, return_counts=True)

    # store the categorical values inside the array
    y = []

    for i in range(len(cat)):
        val = np.where(values == cat[i])
        y.append(val[0][0])
        store = np.asarray(y)

    return store

class DecisionTreeModel:
    # Basic parameter for main classes. Especially the stooping criteria for the tree
    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        # storing the parameters
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series): # We call on the _fit method here
        # TODO
        # call the _fit method
        X = X.to_numpy() # converting x data frame to numphy array
        y = categorical(y) # converting y data frame to numphy array
        self._fit(X, y)
        # end TODO
        print("Done fitting\n") # after the fitting is complete, display this message

    def predict(self, X: pd.DataFrame): # we will call on the _predict method here
        # TODO
        # call the predict method
        X_array = X.to_numpy() # converting x data frame to numphy array
        predict = self._predict(X_array)
        # return ...
        return predict
        # end TODO
        
    def _fit(self, x, y): # fit method will invoke our core method to build the tree
        self.root = self._build_tree(x, y)
        
    def _predict(self, X): # compare the node feature and threshold and decide if we have to take a left or a right turn
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth): # this method is used to evaluate the stopping criteria of the decision tree
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth  # stopping criteria of the tree
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False  # if the stopping criteria is not satisfied then continue with the tree
    
    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X, y, depth=0):  # core method of building the tree
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    

    def _gini(self, y): # calculating gini index from 0 to 0.5
        #TODO
        gini = 0
        proportions = np.bincount(y) / len(y)
        gini = 1 - np.sum([p ** 2 for p in proportions if p > 0])
        #end TODO
        return gini
    
    def _entropy(self, y): # calculating entropy from 0 to 1. Less accurate
        # TODO: the following won't work if y is not integer
        # make it work for the cases where y is a categorical variable
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        # end TODO
        return entropy
        
    def _create_split(self, X, thresh): # helper splitting function the tree into two sides. Left size and the right size
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh): # with this method the tree can be chosen to either gini index or entrophy
        # TODO: fix the code so it can switch between the two criterion: gini and entropy
        parent_loss = self._entropy(y)
        parent_loss = self._gini(y)

        # generating split threshold
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        if self.criterion == 'gini': # setting up the gini index criterion
            child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])

        elif self.criterion == 'entropy': # setting up the entropy criterion
            child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])

        else:  # if the criterion is not satisfied
            print('Please choose a valid criterion')

        # end TODO
        return parent_loss - child_loss
       
    def _best_split(self, X, y, features):
        '''TODO: add comments here
        The tree will loop through all the feature indices and unique threshold values to calculate the information gain.
        Once the information is gained for the specific feature-threshold combination, the tree will compare the result
        to our previous iterations. If the tree finds a better split, we will store it in the associated parameter
        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        # greedy search
        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        '''TODO: add some comments here
        The prediction of the tree will be done by recursively traversing the tree.
        Every data set in the tree will be compared to the current data set and from
        that the tree will decide weather the data goes left or right
        '''
        if node.is_leaf():
            return node.value

        # traversing the tree
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# global method to help with the random forest algorithm
def common(y):
    # calculate the number of occurrence of y
    counter = Counter(y)
    # the number of occurrence are stored in the tuple, but we only need the values
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForestModel(object):

    def __init__(self, n_estimators):
        # n_estimator is the number of tree in the sample
        # TODO:
        pass
        self.tree = [] # store individual decision tree to the empty array
        self.n_estimators = n_estimators
        # end TODO

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        pass
        # training our decision tree to the random forest model
        for _ in range(self.n_estimators):
            tree = DecisionTreeModel(max_depth=10)
            # random sample
            n_samples = X.shape[0]
            index = np.random.choice(n_samples, size=n_samples, replace=True)
            # will raise IndexError if a requested indexer is out-of-bounds,
            # except slice indexers which allow out-of-bounds indexing
            X_sample, y_sample = X.iloc[index], y.iloc[index]

            tree.fit(X_sample, y_sample)
            self.tree.append(tree)

        # once fitting is competed display the message
        print('Done fitting')
        # end TODO
        pass

    def predict(self, X: pd.DataFrame):
        # TODO
        pass
        # Make predictions with every tree in the forest
        tree_preds = np.array([tree.predict(X) for tree in self.tree])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        # return most common label with the tree predictions
        y_pred = [common(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
        # end TODO


def accuracy_score(y_true, y_pred): # calculating the accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    acc = "Accuracy: " + str(accuracy)
    return acc

def classification_report(y_test, y_pred): # I will use the confusion matrix in the next part to calculate the classification report
    # calculate precision, recall, f1-score
    # TODO:

    matrix = np.unique(y_test)
    # initializing the confusion matrix to 0
    result = np.zeros((2, 2))

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # count the number of instances in each combination of actual / predicted classes
            result[i, j] = np.sum((y_test == matrix[i]) & (y_pred == matrix[j]))

    TP = result[0, 0] # True Positive
    FN = result[0, 1] # False Negative
    FP = result[1, 0] # False Positive
    TN = result[1, 1] # True Negative

    # precision formula
    precision = TP / (TP + FP)
    pre = "Precision: " + str(precision)

    # recall formula
    recall = TP / (TP + FN)
    rec = "\nRecall: " + str(recall)

    # F1Score Formula
    x1 = (precision * recall) / (precision + recall)
    F1Score = 2 * x1
    f1 = "\nF1Score: " + str(F1Score) + "\n "

    # end TODO
    return pre + rec + f1

def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    # TODO:
    # extract the different classes
    matrix = np.unique(y_test)

    # initialize the confusion matrix
    result = np.zeros((len(matrix), len(matrix)))

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # count the number of instances in each combination of actual / predicted classes
            result[i, j] = np.sum((y_test == matrix[i]) & (y_pred == matrix[j]))
    # end TODO
    print("Confusion Matrix: ")
    return result

def _test():
    
    df = pd.read_csv("breast_cancer.csv")
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # accuracy
    print(acc)

    # Classification report
    print(classification_report(y_test, y_pred))

    # confusion matrix
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    _test()
