import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# x = np.load("X.npy")
# y = np.load("Y.npy")

# from https://github.com/darkin1/sign-language-digits-ml/blob/master/dataset_fixed.zip
#   from https://www.kaggle.com/ardamavi/sign-language-digits-dataset/discussion/57074

x = np.load("dataset_fixed/X_fixed.npy")
y = np.load("dataset_fixed/Y_fixed.npy")

print(x.shape)
print(y.shape)

n = 5

# for i in range(1, n+1):
#     ax = plt.subplot(1, n, i)
#     plt.imshow(x[i])
#     print(">> ", i, y[i])
#     plt.gray()
#     plt.axis('off')
# plt.show()

inx = int(sys.argv[1])
plt.imshow(x[inx])
plt.title(y[inx])
plt.show()


# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=42)
#
# print("Training set feature matrix shape: " + str(x_train.shape))
# print("Training set classification matrix shape: " + str(y_train.shape))
# print("Testing set feature matrix shape: " + str(x_test.shape))
# print("Testing set classification matrix shape: " + str(y_test.shape))
#
# logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
#
# x_train_rolled = np.reshape(x_train, (x_train.shape[0], -1 ) )
# print(x_train_rolled.shape)
#
# print("test accuracy: {} ".format(logreg.fit(x_train_rolled.T, y_train.T).score(x_test.T, y_test.T)))