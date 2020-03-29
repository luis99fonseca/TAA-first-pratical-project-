# testing from the internet
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn import linear_model

x = np.load("dataset_fixed/X_fixed.npy")
y = np.load("dataset_fixed/Y_fixed.npy")

print(x.shape)
print(y.shape)


X = np.concatenate((x[204:409],x[822:1027]),axis=0)
# Now,we need to create label of zeros and ones.After that we concatenate them.
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z,o),axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.15,random_state=42)
number_of_train = x_train.shape[0]
number_of_test = y_test.shape[0]
print(x_train.shape)
print(y_train.shape)

x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test.reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
print("x train flatten",x_train_flatten.shape)
print("x test flatten",x_test_flatten.shape)

x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_train = y_train.T
y_test = y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))