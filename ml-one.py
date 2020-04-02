import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import time

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

inx = int(sys.argv[1])
plt.imshow(x[inx])
plt.title(y[inx])
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=44)

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

sys.exit(-1)
inx = 1
plt.imshow(x_train[inx])
plt.title(y_train_rolled[inx])
plt.show()

# x train:  (4096, 348)
# x test:  (4096, 62)
# y train:  (1, 348)
# y test:  (1, 62)

########################
# logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 5000) #solver='lbfgs', penalty='l2'
# temp_obj = logreg.fit(x_train_rolled, y_train_rolled)
# print("train accuracy: {} ".format(temp_obj.score(x_train_rolled, y_train_rolled)  ))
# print("test accuracy: {} ".format(temp_obj.score(x_test_rolled, y_test_rolled)  ))
#
# print(">>> ", x_train_rolled[0].shape)
# t_t_r = np.reshape(x_train_rolled[0], (1, x_train_rolled[0].shape[0]))  # shape[0] porque ja retirei um deles, ao ir busca-lo
# t_ts_r = np.reshape(x_test_rolled[0], (1, x_test_rolled[0].shape[0]))
#
# print("predictions train (", y_train_rolled[0], "); got ", temp_obj.predict(t_t_r))
# print("predictions testing (", y_test_rolled[0], "); got ", temp_obj.predict(t_ts_r))
#
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#
# logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 5000, solver='lbfgs', penalty='l2')
# temp_obj = logreg.fit(x_train_rolled, y_train_rolled)
# print("train accuracy: {} ".format(temp_obj.score(x_train_rolled, y_train_rolled)  ))
# print("test accuracy: {} ".format(temp_obj.score(x_test_rolled, y_test_rolled)  ))
#
# print(">>> ", x_train_rolled[0].shape)
# t_t_r = np.reshape(x_train_rolled[0], (1, x_train_rolled[0].shape[0]))  # shape[0] porque ja retirei um deles, ao ir busca-lo
# t_ts_r = np.reshape(x_test_rolled[0], (1, x_test_rolled[0].shape[0]))
#
# print("predictions train (", y_train_rolled[0], "); got ", temp_obj.predict(t_t_r))
# print("predictions testing (", y_test_rolled[0], "); got ", temp_obj.predict(t_ts_r))
########################

# see: https://stackoverflow.com/questions/35956902/how-to-evaluate-cost-function-for-scikit-learn-logisticregression

if False:
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    penalties = ['l1', 'l2', 'elasticnet', 'none']
    C_rgl = [0.01, 0.03, 0.1, 0.3, 1, 1.3]
    for s in solvers:
        for p in penalties:
            for c in C_rgl:
                try:
                    t1 = time.time()
                    logreg = linear_model.LogisticRegression(random_state=42, max_iter=5000, solver=s, penalty=p, C = c)
                    temp_obj = logreg.fit(x_train_rolled, y_train_rolled)
                    print("GOOD: --------", s, " - ", p, " - ", c, "--------")
                    print("In ", t1 - time.time(), " s")
                    print("train accuracy: {} ".format(temp_obj.score(x_train_rolled, y_train_rolled)))
                    print("test accuracy: {} ".format(temp_obj.score(x_test_rolled, y_test_rolled)))

                    t_t_r = np.reshape(x_train_rolled[0], (
                    1, x_train_rolled[0].shape[0]))  # shape[0] porque ja retirei um deles, ao ir busca-lo
                    t_ts_r = np.reshape(x_test_rolled[0], (1, x_test_rolled[0].shape[0]))

                    print("predictions train (", y_train_rolled[0], "); got ", temp_obj.predict(t_t_r))
                    print("predictions testing (", y_test_rolled[0], "); got ", temp_obj.predict(t_ts_r))
                except:
                    print("ERROR: --------", s, " - ", p, " - ", c, "--------")
                    continue

