## Cleaned from ml-seven

import numpy as np
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm
import time
from sklearn.model_selection import cross_validate, GridSearchCV
import matplotlib.pyplot as plt
import sys


# from https://github.com/darkin1/sign-language-digits-ml/blob/master/dataset_fixed.zip
#   from https://www.kaggle.com/ardamavi/sign-language-digits-dataset/discussion/57074

x = np.load("dataset_fixed/X_fixed.npy")
y = np.load("dataset_fixed/Y_fixed.npy")

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=44, stratify=y)

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

# with CV
# try 1
if False:
    kernels = ['rbf']
    C_rgl =  [0.1, 0.3, 1, 3]
    gammas =  [0.003, 0.01, 0.03, 0.1, 0.3, 1]
    for k in kernels:
        for c in C_rgl:
            for g in gammas:
                    t1 = time.time()
                    svmRbg = svm.SVC()
                    parameters = {'C': (c,), 'kernel':(k,), 'gamma':(g,)}
                    gscv = GridSearchCV(svmRbg, param_grid=parameters)
                    gscv.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- c:", c, " - k:", k, " - ", "g:", g)
                    print(">> best params ", gscv.best_params_)
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(gscv.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(gscv.score(x_test_rolled, y_test_rolled)))

                    print("in: ", time.time() - t1)

    # plot_confusion_matrix(gscv, x_test_rolled, y_test_rolled)
    # plt.show()
    # plot_confusion_matrix(gscv, x_train_rolled, y_train_rolled)
    # plt.show()
    # predi = gscv.predict(x_train_rolled)
    # print(classification_report(y_train_rolled, predi))

if True:
    kernels = ['rbf']
    C_rgl =  [3]
    gammas =  [0.009]
    for k in kernels:
        for c in C_rgl:
            for g in gammas:
                    t1 = time.time()
                    svmRbg = svm.SVC()
                    parameters = {'C': (c,), 'kernel':(k,), 'gamma':(g,)}
                    gscv = GridSearchCV(svmRbg, param_grid=parameters)
                    gscv.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- c:", c, " - k:", k, " - ", "g:", g)
                    print(">> best params ", gscv.best_params_)
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(gscv.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(gscv.score(x_test_rolled, y_test_rolled)))

                    print("in: ", time.time() - t1)
print("OUTRO----------------------")
if True:
    kernels = ['rbf']
    C_rgl =  [3]
    gammas =  [0.009]
    for k in kernels:
        for c in C_rgl:
            for g in gammas:
                    t1 = time.time()
                    svmRbg = svm.SVC(kernel=k, gamma=g, C=c)
                    # parameters = {'C': (c,), 'kernel':(k,), 'gamma':(g,)}
                    # gscv = GridSearchCV(svmRbg, param_grid=parameters)
                    svmRbg.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- c:", c, " - k:", k, " - ", "g:", g)
                    # print(">> best params ", gscv.best_params_)
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(svmRbg.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(svmRbg.score(x_test_rolled, y_test_rolled)))

                    print("in: ", time.time() - t1)

sys.exit(-1)
# try 2: tunning
if False:
    kernels = ['rbf']
    C_rgl = [1, 2, 3]
    gammas = [0.008, 0.009, 0.01, 0.012, 0.014]
    for k in kernels:
        for c in C_rgl:
            for g in gammas:
                    t1 = time.time()                                                                                # 5 é o default
                    svmRbg = svm.SVC()
                    parameters = {'C': (c,), 'kernel':(k,), 'gamma':(g,)}
                    gscv = GridSearchCV(svmRbg, param_grid=parameters)
                    gscv.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- c:", c, " - k:", k, " - ", "g:", g)
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(gscv.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(gscv.score(x_test_rolled, y_test_rolled)))

                    print("in: ", time.time() - t1)
# try 3: tunning
elif True:
    kernels = ['rbf']
    C_rgl = [3] #[2, 2.5, 3]
    gammas = [0.009] #[0.01, 0.012, 0.014]
    for k in kernels:
        for c in C_rgl:
            for g in gammas:
                    t1 = time.time()                                                                                # 5 é o default
                    svmRbg = svm.SVC()
                    parameters = {'C': (c,), 'kernel':(k,), 'gamma':(g,)}
                    gscv = GridSearchCV(svmRbg, param_grid=parameters)
                    gscv.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- c:", c, " - k:", k, " - ", "g:", g)
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(gscv.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(gscv.score(x_test_rolled, y_test_rolled)))

                    print("in: ", time.time() - t1)

                    plot_confusion_matrix(gscv, x_test_rolled, y_test_rolled)
                    plt.title("Confusion Matrix SVM: RBF Kernel - Testing")
                    plt.show()
                    plot_confusion_matrix(gscv, x_train_rolled, y_train_rolled)
                    plt.show()
                    predi = gscv.predict(x_test_rolled)
                    print(classification_report(y_test_rolled, predi))
                    print("in: ", time.time() - t1)
                    predi = gscv.predict(x_train_rolled)