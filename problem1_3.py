from sklearn.linear_model import Perceptron
import numpy as np
import sys

data = np.genfromtxt(sys.argv[1], delimiter = ",")
X_train = data[:, :2]
Y_train = data[:, 2]
new_weights = [0, 0, 0]
w = [new_weights]
sum_error = 1

def perceptron_predict(train_row, weights):
    pre = weights[2]
    for i in range(len(train_row)):
        pre += train_row[i]*weights[i]

    return 1 if pre > 0 else -1


    
def perceptron_learning(train, label, weights):
    sum_error = 0
    
    for i in range(len(train)):
        predicted = perceptron_predict(train[i], weights)
        error = label[i] - predicted
        sum_error += error**2
        if predicted*label[i] <= 0:
            weights[2] += int(label[i])
            weights[0] += int(train[i][0]*label[i])
            weights[1] += int(train[i][1]*label[i])
              
    return [weights, sum_error]




