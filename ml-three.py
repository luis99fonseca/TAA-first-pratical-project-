import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import time
from sklearn.metrics import plot_confusion_matrix

# x = np.load("X.npy")
# y = np.load("Y.npy")

# from https://github.com/darkin1/sign-language-digits-ml/blob/master/dataset_fixed.zip
#   from https://www.kaggle.com/ardamavi/sign-language-digits-dataset/discussion/57074

x = np.load("dataset_fixed/X_fixed.npy")
y = np.load("dataset_fixed/Y_fixed.npy")

print(x.shape)
print(y.shape)

number= 5

# for i in range(1, number+1):
#     ax = plt.subplot(1, number, i)
#     plt.imshow(x[i])
#     print(">> ", i, y[i])
#     plt.gray()
#     plt.axis('off')
# plt.show()

# inx = int(sys.argv[1])
# plt.imshow(x[inx])
# plt.title(y[inx])
# plt.show()

test_sizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30]
C_rgl = [0.01, 0.03, 0.1, 0.3, 1, 1.3]
C_rgl = [3]
test_sizes = [0.2]

if True:
    for t in test_sizes:
        for c in C_rgl:

            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=t , random_state=44)
            print("BEGIN--------------------------- t=", t, "; c=", c)
            print("Training set feature matrix shape: " + str(x_train.shape))
            print("Training set classification matrix shape: " + str(y_train.shape))
            print("Testing set feature matrix shape: " + str(x_test.shape))
            print("Testing set classification matrix shape: " + str(y_test.shape))
            print("------------------------------------------------")

            x_train_rolled = np.reshape(x_train, (x_train.shape[0], -1 ) )
            m, n = x_train_rolled.shape
            print("n: ", n, "; m: ", m)
            print(x_train_rolled.shape)
            y_train_rolled = np.reshape(np.argmax(y_train, axis = 1), (m,)) # eles pedem assim
            print("Training set rolled feature matrix shape: ", x_train_rolled.shape)
            print("Training set rolled classification matrix shape: ", y_train_rolled.shape)
            print("------------------------------------------------")
            x_test_rolled = np.reshape(x_test, (x_test.shape[0], -1))
            mt, nt = x_test_rolled.shape
            print("nt: ", nt, "; mt: ", mt)
            y_test_rolled = np.reshape(np.argmax(y_test, axis = 1), (mt,))
            print("Testing set rolled feature matrix shape: ", x_test_rolled.shape)
            print("Testing set rolled classification matrix shape: ", y_test_rolled.shape)


            print(">>> ", y_train_rolled)
            unique, counts = np.unique(y_train_rolled, return_counts=True)
            print(dict(zip(unique, counts)))

            unique, counts = np.unique(y_test_rolled, return_counts=True)
            print(dict(zip(unique, counts)))
            t1 = time.time()
            logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 5000, C = c) #solver='lbfgs', penalty='l2'
            temp_obj = logreg.fit(x_train_rolled, y_train_rolled)
            print("train accuracy: {} ".format(temp_obj.score(x_train_rolled, y_train_rolled)  ))
            print("test accuracy: {} ".format(temp_obj.score(x_test_rolled, y_test_rolled)  ))

            print(plot_confusion_matrix(logreg, x_test_rolled, y_test_rolled))
            plt.show()

            print("END---------------------------", time.time() - t1)

elif False:
    t = 0.2
    c = 0.3

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=t, random_state=44)
    print("BEGIN--------------------------- t=", t, "; c=", c)
    print("Training set feature matrix shape: " + str(x_train.shape))
    print("Training set classification matrix shape: " + str(y_train.shape))
    print("Testing set feature matrix shape: " + str(x_test.shape))
    print("Testing set classification matrix shape: " + str(y_test.shape))
    print("------------------------------------------------")

    x_train_rolled = np.reshape(x_train, (x_train.shape[0], -1))
    m, n = x_train_rolled.shape
    print("n: ", n, "; m: ", m)
    print(x_train_rolled.shape)
    y_train_rolled = np.reshape(np.argmax(y_train, axis=1), (m,))  # eles pedem assim
    print("Training set rolled feature matrix shape: ", x_train_rolled.shape)
    print("Training set rolled classification matrix shape: ", y_train_rolled.shape)
    print("------------------------------------------------")
    x_test_rolled = np.reshape(x_test, (x_test.shape[0], -1))
    mt, nt = x_test_rolled.shape
    print("nt: ", nt, "; mt: ", mt)
    y_test_rolled = np.reshape(np.argmax(y_test, axis=1), (mt,))
    print("Testing set rolled feature matrix shape: ", x_test_rolled.shape)
    print("Testing set rolled classification matrix shape: ", y_test_rolled.shape)

    print(">>> ", y_train_rolled)
    unique, counts = np.unique(y_train_rolled, return_counts=True)
    print(dict(zip(unique, counts)))

    unique, counts = np.unique(y_test_rolled, return_counts=True)
    print(dict(zip(unique, counts)))
    t1 = time.time()

    from sklearn.linear_model.logistic import _logistic_loss

    logreg = linear_model.LogisticRegressionCV(random_state=42, max_iter=5000)  # solver='lbfgs', penalty='l2'
    temp_obj = logreg.fit(x_train_rolled, y_train_rolled)
    print("train accuracy: {} ".format(logreg.score(x_train_rolled, y_train_rolled)))
    print("test accuracy: {} ".format(logreg.score(x_test_rolled, y_test_rolled)))
    print(x_train_rolled.shape, y_train_rolled.shape)
    # print(_logistic_loss(logreg.coef_, x, y, 1 / logreg.C) )
    print("END---------------------------", time.time() - t1)


# inx = 1
# plt.imshow(x_train[inx])
# plt.title(y_train_rolled[inx])
# plt.show()

