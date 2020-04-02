import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import time
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV


# from https://github.com/darkin1/sign-language-digits-ml/blob/master/dataset_fixed.zip
#   from https://www.kaggle.com/ardamavi/sign-language-digits-dataset/discussion/57074

x = np.load("dataset_fixed/X_fixed.npy")
y = np.load("dataset_fixed/Y_fixed.npy")

print(x.shape)
print(y.shape)

number = 5

# Css = [3.2, 5, 10, 40, 100]
# gammas = [0.01, 0.1, 1, 10, 100, 1000, "scale", "auto"]
# kernels = ['rbf'] #, 'linear', 'sigmoid', 'precomputed', 'poly']
# Css = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# gammas = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

## RBF Tunning

Css = [10]
gammas = [0.01, 0.0065, 0.006, 0.0055]
kernels = ['rbf']

coords1 = []
coords2 = []
# with open("resultsNew.txt", "w") as file01:
if True:
    if True:
        for c in Css:
            for g in gammas:
                for k in kernels:
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
                    print("BEGIN--------------------------- c=",c, "; g=",g, "; k=",k)
                    # print("Training set feature matrix shape: " + str(x_train.shape))
                    # print("Training set classification matrix shape: " + str(y_train.shape))
                    # print("Testing set feature matrix shape: " + str(x_test.shape))
                    # print("Testing set classification matrix shape: " + str(y_test.shape))
                    # print("------------------------------------------------")

                    x_train_rolled = np.reshape(x_train, (x_train.shape[0], -1))
                    m, n = x_train_rolled.shape
                    print("n: ", n, "; m: ", m)
                    print(x_train_rolled.shape)
                    y_train_rolled = np.reshape(np.argmax(y_train, axis=1), (m,))  # eles pedem assim
                    # print("Training set rolled feature matrix shape: ", x_train_rolled.shape)
                    # print("Training set rolled classification matrix shape: ", y_train_rolled.shape)
                    # print("------------------------------------------------")
                    x_test_rolled = np.reshape(x_test, (x_test.shape[0], -1))
                    mt, nt = x_test_rolled.shape
                    # print("nt: ", nt, "; mt: ", mt)
                    y_test_rolled = np.reshape(np.argmax(y_test, axis=1), (mt,))
                    # print("Testing set rolled feature matrix shape: ", x_test_rolled.shape)
                    # print("Testing set rolled classification matrix shape: ", y_test_rolled.shape)

                    # print(">>> ", y_train_rolled)
                    unique, counts = np.unique(y_train_rolled, return_counts=True)
                    print(dict(zip(unique, counts)))

                    unique, counts = np.unique(y_test_rolled, return_counts=True)
                    print(dict(zip(unique, counts)))
                    t1 = time.time()

                    # print("gamma: ", 1 / (x_train_rolled.shape[1] * X.var()))
                    # sys.exit(-1)
                    logreg = svm.SVC(C=c, kernel=k, gamma=g)
                    # temp_obj = logreg.fit(x_train_rolled, y_train_rolled)
                    # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                    #               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
                    # logreg = GridSearchCV(
                    #     svm.SVC(kernel='rbf', class_weight='balanced'), param_grid
                    # )
                    temp_obj = logreg.fit(x_train_rolled, y_train_rolled)
                    # file01.write("---------- c=" + str(c) + "; g=" + str(g) + "; k=" + str(k) + "\n")
                    # file01.write("train accuracy: {} ".format(temp_obj.score(x_train_rolled, y_train_rolled)) + "\n")
                    # file01.write("test accuracy: {} ".format(temp_obj.score(x_test_rolled, y_test_rolled)) + "\n")
                    # file01.write("end ----------" + "\n")
                    print("train accuracy: {} ".format(temp_obj.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(temp_obj.score(x_test_rolled, y_test_rolled)))
                    coords1.append(temp_obj.score(x_train_rolled, y_train_rolled))
                    coords2.append(temp_obj.score(x_test_rolled, y_test_rolled))

                    # print(plot_confusion_matrix(logreg, x_test_rolled, y_test_rolled))
                    # plt.show()

                    print("END---------------------------", time.time() - t1)

plt.plot(gammas, coords1, 'r', gammas, coords2, 'b')
plt.show()