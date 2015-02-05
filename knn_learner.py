"""
Performs K-nearest neighbor learning on a given dataset.
"""

import sys #for command-line arguments
from sklearn import neighbors # k-nearest neighbor learning
import numpy # processing
import random # for splitting data in training/test

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print('Usage: python knn_learner <train_data> [test_data]')
        return
    
    # Have to treat data differently if we have separate testing data or not
    has_test_data = (len(sys.argv) == 3)
    
    # Open the training file and get the categories
    train_file = open(sys.argv[1], 'r')
    categories = train_file.readline().replace(',', '').split()
    
    # Split training and testing data.
    if not has_test_data:
        all_data = numpy.loadtxt(train_file)
        rows, columns = all_data.shape
        train_data = numpy.empty([0, columns])
        test_data = numpy.empty([0, columns])
        test_data_labels = numpy.empty([0, 1])
        train_data_labels = numpy.empty([0, 1])
        for i in range(rows):
            if random.random() > 0.7:
                test_data.append(all_data[i,0:(columns-2)], axis=0)
                test_data_labels.append(all_data[i,columns-1], axis=0)
            else:
                train_data.append(all_data[i,0:(columns-2)], axis=0)
                train_data_labels.append(all_data[i,columns-1], axis=0)
    else:
        train_data = numpy.loadtxt(train_file)
        rows, columns = train_data.shape
        train_data_labels = train_data[:,columns-1]
        train_data = train_data[:,0:(columns-2)
        test_file = open(sys.argv[2], 'r')
        test_data = numpy.loadtxt(test_file)
        rows, columns = test_data.shape
        test_data_labels = test_data[:,columns-1]
        test_data = test_data[:,0:(columns-2)
        
    k_neighbors = [1, 5, 15]

    kneighbors_predict = []
    kneighbors_check = []

    for num_neighbors in k_neighbors:
        kneighbors = neighbors.KNeighborsClassifier(num_neighbors)
        kneighbors.fit(train_data, train_data_labels)
        kneighbors_predict.append(kneighbors.predict(test_data))
        kneighbors_check.append(kneighbors.predict(train_data))
    
    return


main()