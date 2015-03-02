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
    
def knn_learner(train_data, train_labels, k_vals):
    """
    Performs K-nearest neighbors on the test data.
    """
    kneighbors_classifiers = []

    for num_neighbors in k_vals:
        kneighbors = neighbors.KNeighborsClassifier(num_neighbors)
        kneighbors.fit(train_data, train_labels)
        kneighbors_classifiers.append(kneighbors)
    
    return kneighbors_classifiers

def count_differences(vec1, vec2):
    """
    Finds the total number of elements at which vector 1 differs from vector 2.
    """
    diff_vec = vec1 - vec2
    return numpy.linalg.norm(numpy.absolute(diff_vec), 1)

def print_check_dataset(classifier_name, classifier, data_name, data, labels):
    predict = classifier.predict(data)
    num_total_points = len(predict)
    num_wrong = count_differences(predict, labels)
    print(classifier_name + ": On " + data_name + " data, a total of "
          + str(num_wrong) + " points were predicted wrong of "
          + str(num_total_points) + ".")
    percent_right = 100*(num_total_points - num_wrong)/num_total_points
    print("This is an accuracy of " + str(percent_right) + "%")

def check_classifier(classifier_name, classifier, datasets):
    (train_data, train_labels, test_data, test_labels) = datasets
    check_dataset(classifier_name, classifier, "training",
                  train_data, train_labels)
    
    check_dataset(classifier_name, classifier, "testing",
                  test_data, test_labels)

def predict_percent(classifier, test_data, test_labels):
    predict = classifier.predict(test_data)
    num_total_points = len(predict)
    num_wrong = count_differences(predict, test_labels)
    percent_right = 100*(num_total_points - num_wrong)/num_total_points
    
    return (num_total_points, num_wrong, percent_right)
    
def get_absolute_index(index, indices_removed, total_indices):
    """
    Determines the absolute index of a given index in a subset of an original
    array which has had certain indices removed.
    
    Ex. in the array [0, 1, 2, 3, 4, 5, 6]
    if we remove indices 3 and 5, we get the new subarray [0, 1, 2, 4, 6].
    Index 3 in this has value 4, and index 4 in the original array. We
    want to know that absolute index 4, given index 3, indices_removed [3, 5],
    and total_indices 7.
    """
    absolute_index = index
    for removed in indices_removed.sort():
        if removed <= absolute_index:
            absolute_index += 1
    
    return absolute_index
    
def backward_eliminate(which_learner, datasets):
    print("Backward elimination is a TODO.")
    return
    # Split dataset into components
    (train_data, train_labels, test_data, test_labels) = datasets
    
    # Create classifier based on which learner was used
    if(which_learner == "KNN"):
        classifier = knn_learner(train_data, train_labels, [5])[0]
    elif(which_learner == "Linear"):
        classifier = linear_learner(train_data, train_labels)
    else:
        return
    
    max_num_features = train_data.shape[1]
    
    # General info for iterating
    predict_info = predict_percent(classifier, test_data, test_labels)
    (num_total_points, num_wrong, best_percent_right) = predict_info
    indices_removed = []
    final_data = test_data
    
    # Iterate until there is one feature left
    for i in range(max_num_features):
        
        best_percent_tmp = 0
        feature_removed_tmp = -1
        for j in range(max_num_features - i):
            #Remove a feature from the remaining features
            #print('TODO! Backwards greedy elimination.')
            #break
            #TODO: I need to be updating the training data as I go.
            
            #Remove a feature and test it
            tmp_data = numpy.delete(final_data, j, 1)
            predict_info = predict_percent(classifier, tmp_data, test_labels)
            percent_tmp = predict_info[2]
            
            #If this is an improvement, keep track of it
            if(percent_tmp > best_percent_tmp):
                feature_removed_tmp = get_absolute_index(j, indices_removed,
                                                        max_num_features)
                best_percent_tmp = percent_tmp
                best_data_tmp = tmp_data
            
        
        # If not improved, stop 
        if best_percent_tmp < best_percent_right:
            break
        
        # Otherwise, keep track and keep going
        indices_removed.append(feature_removed_tmp)
        final_data = best_data_tmp
    
    
    #feature_selector = feature_selection.RFECV(classifier)
    #feature_selector.fit(train_data, train_labels)
    #print(feature_selector.get_support())
    
    #check_classifier(which_learner + " with selection", feature_selector,
    #                 datasets)

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('Usage: python binary_learner <train_data> [test_data]')
        return

    train_filename = sys.argv[1]
    test_filename = sys.argv[2] if len(sys.argv) == 3 else ''
    
    datasets = split_data(train_filename, test_filename)
    (train_data, train_labels, test_data, test_labels) = datasets
    
    # knn_classifier = knn_learner(train_data, train_labels, [5])
    #check_classifier("KNN", knn_classifier[0], datasets)
    
    #linear_classifier = linear_learner(train_data, train_labels)
    #check_classifier("Linear", linear_classifier, datasets)
    
    backward_eliminate("Linear", datasets)
    
    

main()
