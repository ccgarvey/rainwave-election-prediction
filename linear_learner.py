"""
Performs K-nearest neighbor learning on a given dataset.
"""

import sys #for command-line arguments
from sklearn import linear_model # linear regression learning
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
        all_data = numpy.loadtxt(train_file, delimiter=', ')
        rows, columns = all_data.shape
        train_data = numpy.zeros([0, columns])
        train_data_labels = numpy.zeros([0, 1])
        train_rows = 0
        test_data = numpy.zeros([0, columns])
        test_data_labels = numpy.zeros([0, 1])
        test_rows = 0
        
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
        
    else:
        train_data = numpy.loadtxt(train_file)
        rows, columns = train_data.shape
        train_data_labels = train_data[:,columns-1]
        train_data = train_data[:,0:(columns-2)]
        test_file = open(sys.argv[2], 'r')
        test_data = numpy.loadtxt(test_file)
        rows, columns = test_data.shape
        test_data_labels = test_data[:,columns-1]
        test_data = test_data[:,0:(columns-2)]
    

    linear = linear_model.LinearRegression()
    linear.fit(train_data, train_data_labels)
    linear_predict = linear.predict(test_data)
    linear_check = linear.predict(train_data)
    
    return


main()