import numpy as np
import matplotlib.pyplot as plt

# with stratify
if True:
    Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 6, 10]
    training = np.array([
        0.8926622195269861,
        0.9563371740448757,
        0.9963614311704063,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ])

    testing = np.array([
        0.7820823244552058,
        0.8159806295399515,
        0.8135593220338984,
        0.8159806295399515,
        0.8159806295399515,
        0.8159806295399515,
        0.8159806295399515,
        0.8159806295399515
    ])
    plt.plot(Cs, training, '-o', Cs, testing, '-o')
    plt.legend(["Training", "Testing"])
    plt.xlabel("Inverse of Regularization Strength")
    plt.ylabel("Accuracy")
    plt.title("SVM - Linear Kernel")
    plt.show()

# tunning
elif True:
    Cs = [0.02, 0.03, 0.4]
    training = np.array([
        0.9357186173438448,
        0.9563371740448757,
        1.0
    ])

    testing = np.array([
        0.8087167070217918,
        0.8159806295399515,
        0.8159806295399515,
    ])
    plt.plot(Cs, training, '-o', Cs, testing, '-o')
    plt.legend(["Training", "Testing"])
    plt.xlabel("Inverse of Regularization Strength")
    plt.ylabel("Accuracy")
    plt.title("SVM - Linear Kernel - tunning")
    plt.show()