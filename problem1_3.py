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


    

