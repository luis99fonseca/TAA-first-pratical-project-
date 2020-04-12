## Cleaned from ml-seven

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import time

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
if True:
    solvers = ['newton-cg', 'lbfgs']
    penalties = ['l2']
    C_rgl = [0.01, 0.03, 0.1, 0.3, 1, 1.3, 3, 6, 10]
    for s in solvers:
        for p in penalties:
            # for c in C_rgl:
                    t1 = time.time()                                                                                # 5 é o default
                    logreg = linear_model.LogisticRegressionCV(random_state=42, max_iter=15000, solver=s, penalty=p, cv=5, Cs=C_rgl)
                    logreg.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- s:", s, " - p:", p, " - ", "c:", "c")
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(logreg.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(logreg.score(x_test_rolled, y_test_rolled)))

                    # print("scores2: ", logreg.scores_)

                    print("Cs2: ", logreg.Cs_)
                    print("best C: ", logreg.C_)

                    print("in: ", time.time() - t1)

# with CV, with K tunning
elif False:
    solvers = ['newton-cg', 'lbfgs']
    penalties = ['l2']
    C_rgl = [0.1]
    print("On K tunning")
    for s in solvers:
        for p in penalties:
            for c in C_rgl:
                for k in range(2,11):
                    t1 = time.time()
                    logreg = linear_model.LogisticRegressionCV(random_state=42, max_iter=15000, solver=s, penalty=p, cv=k, Cs=[c])
                    logreg.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- s:", s, " - p:", p, " - ", "c:", c, " - k:", k)
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(logreg.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(logreg.score(x_test_rolled, y_test_rolled)))

                    print("scores2: ", logreg.scores_)

                    print("Cs2: ", logreg.Cs_)
                    print("best C: ", logreg.C_)

                    print("in: ", time.time() - t1)

# with CV, tunning 2
elif True:
        solvers = ['newton-cg', 'lbfgs']
        penalties = ['l2']
        C_rgl = [0.08, 0.1, 0.12]
        for s in solvers:
            for p in penalties:
                for c in C_rgl:
                    t1 = time.time()  # 5 é o default
                    logreg = linear_model.LogisticRegressionCV(random_state=42, max_iter=15000, solver=s, penalty=p,
                                                               cv=5, Cs=[c])
                    logreg.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: -------- s:", s, " - p:", p, " - ", "c:", c)
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(logreg.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(logreg.score(x_test_rolled, y_test_rolled)))

                    print("scores2: ", logreg.scores_)

                    print("Cs2: ", logreg.Cs_)
                    print("best C: ", logreg.C_)

                    print("in: ", time.time() - t1)
