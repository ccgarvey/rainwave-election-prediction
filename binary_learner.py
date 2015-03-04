"""
Performs binary learning using K-nearest neighbors or linear regression.
"""

import sys #for command-line arguments
from sklearn import linear_model # linear regression learning
from sklearn import neighbors # k-nearest neighbors
from sklearn import feature_selection # Recursive feature elimination
import numpy # processing
import random # for splitting data in training/test


def split_data(train_filename, test_filename):
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
    Performs linear regression on the testing data.
    
    """
    linear = linear_model.LinearRegression()
    linear.fit(train_data, train_labels)
    
    return linear
    
def knn_regress_learner(train_data, train_labels, k_vals, datasets):
    """
    Performs K-nearest neighbors on the test data.
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

def count_differences(vec1, vec2):
    """
    Finds the total number of elements at which vector 1 differs from vector 2.
    """
    diff_vec = vec1 - vec2
    return numpy.linalg.norm(numpy.absolute(diff_vec), 1)

def check_dataset(classifier_name, classifier, data_name, data, labels):
    predict = classifier.predict(data)
    num_total_points = len(predict)
    num_wrong = count_differences(predict, labels)
    print(classifier_name + ": On " + data_name + " data, a total of "
          + str(num_wrong) + " points were predicted wrong of "
          + str(num_total_points) + ".")
    percent_right = 100*(num_total_points - num_wrong)/num_total_points
    print(classifier_name + " This is an accuracy of " + str(percent_right) + "%")

def check_classifier(classifier_name, classifier, datasets):
    (train_data, train_labels, test_data, test_labels) = datasets
    check_dataset(classifier_name, classifier, "training",
                  train_data, train_labels)
    
    check_dataset(classifier_name, classifier, "testing",
                  test_data, test_labels)
    
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
                     datasets)
    
    
    
def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('Usage: python binary_learner <train_data> [test_data]')
        return

    train_filename = sys.argv[1]
    test_filename = sys.argv[2] if len(sys.argv) == 3 else ''
    
    datasets = split_data(train_filename, test_filename)
    (train_data, train_labels, test_data, test_labels) = datasets
    
    knn_values = range(1, 15)
    knn_classifier = knn_learner(train_data, train_labels, knn_values, datasets)
    knr_classifier = knn_regress_learner(train_data, train_labels, knn_values, datasets)
    for i in range(len(knn_values)):
        check_classifier("KNN", knn_classifier[i], datasets)
        check_classifier("KNR", knr_classifier[i], datasets)
    
    #linear_classifier = linear_learner(train_data, train_labels)
    #check_classifier("Linear", linear_classifier, datasets)
    
    #subset_select("Linear", datasets)
    
    

main()
