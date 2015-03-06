"""
Performs binary learning using K-nearest neighbors or linear regression.
"""

import sys #for command-line arguments
from sklearn import linear_model # linear regression learning
from sklearn import neighbors # k-nearest neighbors
from sklearn import feature_selection # Recursive feature elimination
from sklearn.decomposition import PCA # Principal component analysis
from sklearn import preprocessing # mean 0 variance 1
from sklearn import tree # binary decision tree
import numpy # processing
from scipy import stats # thresholding
import random # for splitting data in training/test



def split_data(train_filename, test_filename, keep_elections=False):
    """
    If there is no separate train/test file, test_filename has length 0.
    
    return: test data features
    return: test data labels
    return: training data features
    return: training data labels
    """
    # Have to treat data differently if we have separate testing data or not
    has_test_data = (len(test_filename) > 0)
    
    # Open the training file and get the categories
    train_file = open(train_filename, 'r')
    categories = train_file.readline().replace(',', '').split()
    
    # Split training and testing data.
    if not has_test_data:
        all_data = numpy.loadtxt(train_file, delimiter=', ')
        rows, columns = all_data.shape
        train_data = numpy.zeros([0, columns])
        train_data_labels = numpy.zeros([0, 1])
        train_rows = 0
        test_data = numpy.zeros([0, columns])
        test_data_labels = numpy.zeros([0, 1])
        test_rows = 0
        
        # Randomly select training and testing data from file(s)
        if keep_elections:
            #Split the data so that each election is maintained in a data set
            cur_row = 0
            while cur_row < rows:
                if random.random() > 0.7:
                    at_first = True
                    while (cur_row < rows) and (at_first or \
                                             all_data[cur_row, columns-1] == 0):
                        test_data = numpy.append(test_data,
                                                all_data[cur_row,:(columns-1)])
                        test_data_labels = numpy.append(test_data_labels,
                                                        all_data[cur_row,
                                                                 columns-1])
                        at_first = False
                        test_rows += 1
                        cur_row += 1
                else:
                    at_first = True
                    while (cur_row < rows) and (at_first or \
                                             all_data[cur_row, columns-1] == 0):
                        train_data = numpy.append(train_data,
                                                all_data[cur_row,:(columns-1)])
                        train_data_labels = numpy.append(train_data_labels,
                                                        all_data[cur_row,
                                                                 columns-1])
                        at_first = False
                        train_rows += 1
                        cur_row += 1
        
        else:
            for i in range(rows):
                if random.random() > 0.7:
                    test_data = numpy.append(test_data, all_data[i,:(columns-1)])
                    test_data_labels = numpy.append(test_data_labels,
                                                    all_data[i,columns-1])
                    test_rows += 1
                else:
                    train_data = numpy.append(train_data, all_data[i,0:(columns-1)])
                    train_data_labels = numpy.append(train_data_labels,
                                                    all_data[i,columns-1])
                    train_rows += 1
        
        test_data = test_data.reshape((test_rows, columns-1))
        train_data = train_data.reshape((train_rows, columns-1))
    
    # Training and testing files are specified already
    else:
        # Get the training data
        train_data = numpy.loadtxt(train_file, delimiter=', ')
        rows, columns = train_data.shape
        train_data_labels = train_data[:,columns-1]
        train_data = train_data[:,:(columns-1)]
        
        # Get the testing data
        test_file = open(test_filename, 'r')
        test_data = numpy.loadtxt(test_file, delimiter=', ')
        rows, columns = test_data.shape
        test_data_labels = test_data[:,columns-1]
        test_data = test_data[:,:(columns-1)]

    return (train_data, train_data_labels, test_data, test_data_labels)

def linear_learner(train_data, train_labels):
    """
    Performs linear regression on the training data.
    
    """
    linear = linear_model.LinearRegression()
    linear.fit(train_data, train_labels)
    
    return linear

def binary_tree_learner(train_data, train_labels):
    """
    Creates a binary tree learner based on the training data.
    """
    bin_tree = tree.DecisionTreeClassifier(max_depth=1)
    bin_tree.fit(train_data, train_labels)
    #print(bin_tree.feature_importances_)
    
    return bin_tree 
    
def knn_regress_learner(train_data, train_labels, k_vals, datasets):
    """
    Performs K-nearest neighbors on the training data.
    """
    kneighbors_classifiers = []

    for num_neighbors in k_vals:
        kneighbors = neighbors.KNeighborsRegressor(n_neighbors=num_neighbors)
        kneighbors.fit(train_data, train_labels)
        kneighbors_classifiers.append(kneighbors)
    
    return kneighbors_classifiers

def knn_learner(train_data, train_labels, k_vals, datasets):
    """
    Performs K-nearest neighbors regression on the test data.
    """
    kneighbors_classifiers = []

    for num_neighbors in k_vals:
        kneighbors = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
        kneighbors.fit(train_data, train_labels)
        kneighbors_classifiers.append(kneighbors)
    
    return kneighbors_classifiers

def threshold_values(vector):
    """
    Returns a new vector with all values < 0.5 set to 0, and all values >= 0.5
    set to 1.
    """
    tmp_vec = stats.threshold(vector, None, 0.5, 1)
    tmp_vec = stats.threshold(tmp_vec, 0.5, None, 0)
    return tmp_vec
    
def pick_best(values):
    """
    Selects the index in values that has the highest value. Returns it.
    If there is more than one value that has the same value, returns a random
    choice of those indices.
    """
    max_value = max(values)
    count = 0
    for i in values:
        if i == max_value:
            count += 1
            
    which_value = guess(count)
    index = 0
    for i in values:
        if i == max_value:
            if which_value == 0:
                return index
            else:
                which_value -= 1
        index += 1
    return -1
    

def elections_correct_percent(classifier, data, labels, should_guess=False):
    if not should_guess:
        predict = classifier.predict(data)
    num_total_points = len(labels)
    
    point = 0
    elections = 0
    num_elections_wrong = 0
    while point < num_total_points:
        # Check who the predicted winner of the election is
        at_first = True
        values = []
        count = 0
        while (point < num_total_points) and (at_first or labels[point] == 0):
            at_first = False
            if not should_guess:
                values.append(predict[point])
            else:
                count += 1
            point += 1
        
        # this may count a tie wrong; this is intentional.
        #if not should_guess:
        #    print(values)
        #print(labels[(point-3):(point):1])
        if not should_guess:
            max_point = pick_best(values)
        else:
            max_point = guess(count)
        #max_point = (len(values) - values.index(max(reversed(values)))) - 1
        #print(max_point)
        if(labels[max_point] != 1):
            num_elections_wrong += 1
        elections += 1

    percent_right = 100*(elections - num_elections_wrong)/elections
    return (elections, num_elections_wrong, percent_right)
    
def songs_correct_percent(classifier, test_data, test_labels,  should_threshold=False):
    """
    """
    predict = classifier.predict(test_data)
    if(should_threshold):
        predict = threshold_values(predict)
    num_total_points = len(predict)
    num_wrong = count_differences(predict, test_labels)
    percent_right = 100*(num_total_points - num_wrong)/num_total_points
    
    return (num_total_points, num_wrong, percent_right)

def count_differences(vec1, vec2):
    """
    Finds the total number of elements at which vector 1 differs from vector 2.
    """
    diff_vec = vec1 - vec2
    return numpy.linalg.norm(numpy.absolute(diff_vec), 1)

def check_dataset(classifier_name, classifier, data_name, data, labels, guess):
    (num_total_points, num_wrong, percent_right) = \
        elections_correct_percent(classifier, data, labels, guess)
        #songs_correct_percent(classifier, data, labels, True)
        
        
    print(classifier_name + ": On " + data_name + " data, a total of "
          + str(num_wrong) + " points were predicted wrong of "
          + str(num_total_points) + ".")
    percent_right = 100*(num_total_points - num_wrong)/num_total_points
    print(classifier_name + ": This is an accuracy of " + str(percent_right) + "%")

def check_classifier(classifier_name, classifier, datasets, guess=False):
    (train_data, train_labels, test_data, test_labels) = datasets
    #check_dataset(classifier_name, classifier, "training",
    #              train_data, train_labels)
    
    check_dataset(classifier_name, classifier, "testing",
                  preprocessing.scale(test_data), test_labels, guess)
    
def subset_select(which_learner, datasets):
    """
    Uses recursive feature elimination to select the best features.
    """
    # Split dataset into components
    (train_data, train_labels, test_data, test_labels) = datasets
    
    # Create classifier based on which learner was used
    if(which_learner == "KNN"):
        classifier = knn_learner(train_data, train_labels, [5])[0]
    elif(which_learner == "Linear"):
        classifier = linear_learner(train_data, train_labels)
    else:
        return
    
    feature_selector = feature_selection.RFECV(classifier)
    feature_selector.fit(train_data, train_labels)
    print(feature_selector.get_support())
    
    check_classifier(which_learner + " with selection", feature_selector,
                     datasets, True)
    
def doPCA(data, other_data):
    scaled = preprocessing.scale(data)
    #analyzer = PCA()
    analyzer = PCA(n_components=5)
    analyzer.fit(scaled)
    #print(analyzer.components_)
    #print(analyzer.explained_variance_ratio_)
    #print(analyzer.get_covariance())
    return analyzer.transform(data), analyzer.transform(other_data)
    
def guess(num_songs_in_election):
    return random.randint(0, num_songs_in_election-1)
    
def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('Usage: python binary_learner <train_data> [test_data]')
        return

    train_filename = sys.argv[1]
    test_filename = sys.argv[2] if len(sys.argv) == 3 else ''
    
    datasets = split_data(train_filename, test_filename, True)
    (train_data, train_labels, test_data, test_labels) = datasets
    #train_data, test_data = doPCA(train_data, test_data)
    #datasets = (train_data, train_labels, test_data, test_labels)
    
    #knn_values = range(1, 15)
    knn_values = [10]
    knn_classifier = knn_learner(preprocessing.scale(train_data), train_labels, knn_values, datasets)
    knr_classifier = knn_regress_learner(train_data, train_labels, knn_values, datasets)
    for i in range(len(knn_values)):
        check_classifier("KNN", knn_classifier[i], datasets)
        check_classifier("KNR", knr_classifier[i], datasets)
    
    tree_classifier = binary_tree_learner(train_data, train_labels)
    check_classifier("Decision tree", tree_classifier, datasets)
    
    linear_classifier = linear_learner(train_data, train_labels)
    check_classifier("Linear", linear_classifier, datasets)
    
    #subset_select("Linear", datasets)
    
    check_classifier("Guess", None, datasets, True)

main()
