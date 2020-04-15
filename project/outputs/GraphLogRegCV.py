import numpy as np
import matplotlib.pyplot as plt

# without stratify
if False:
    if True:
        ## Method: LogReg
        ### Solver: Newton-cg
        Cs = [0.01, 0.03, 0.1, 0.3, 1, 1.3, 3]
        training = np.array([
            0.8283808368708309,
            0.8884172225591267,
            0.9526986052152819,
            0.9927228623408126,
            1.0,
            1.0,
            1.0
        ])

        testing = np.array([
            0.6949152542372882,
            0.7215496368038741,
            0.7360774818401937,
            0.7336561743341404,
            0.7312348668280871,
            0.7288135593220338,
            0.7191283292978208
        ])
        plt.plot(Cs,training, '-o', Cs, testing, '-o')
        # plt.legend(["Training", "Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")


        ### Solver: Lbfgs
        Cs = [0.01, 0.03, 0.1, 0.3, 1, 1.3, 3]
        training = [            # sem CV
            0.8283808368708309,
            0.8890236506973923, # 0.8878107944208611
            0.9526986052152819, # 0.9533050333535477
            0.9927228623408126,
            1.0,
            1.0,
            1.0
        ]
        testing = [
            0.6949152542372882,
            0.7215496368038741,
            0.7360774818401937,
            0.7336561743341404,
            0.7312348668280871,
            0.7263922518159807,
            0.7191283292978208
        ]

        plt.plot(Cs,training, '-o', Cs, testing, '-o')
        plt.legend(["N-cg Training","N-cg Testing"] + ["Lbfgs Training", "Lbfgs Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")
        plt.show()

    # tunning
    if False:
        ## Method: LogReg
        ### Solver: Newton-cg
        Cs = [0.08, 0.1, 0.12]
        training = np.array([
            0.939963614311704,
            0.9526986052152819,
            0.9617950272892662,

        ])

        testing = np.array([
            0.738498789346247,
            0.7360774818401937,
            0.7312348668280871,
        ])
        plt.plot(Cs,training, '-o', Cs, testing, '-o')
        # plt.legend(["Training", "Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")


        ### Solver: Lbfgs
        Cs = [0.08, 0.1, 0.12]
        training = [
            0.939963614311704,
            0.9526986052152819,
            0.9617950272892662,
        ]
        testing = [
            0.738498789346247,
            0.7360774818401937,
            0.7312348668280871,
        ]

        plt.plot(Cs,training, '-o', Cs, testing, '-o')
        # plt.title("T1")
        plt.legend(["N-cg Training","N-cg Testing"] + ["Lbfgs Training", "Lbfgs Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")
        plt.show()

# with stratify
if True:
    if False:
        ## Method: LogReg
        ### Solver: Newton-cg
        Cs = [0.01, 0.03, 0.1, 0.3, 1, 1.3, 3, 6, 10]
        training = np.array([
            0.8211036992116434,
            0.8835657974530018,
            0.9496664645239539,
            0.9909035779260158,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ])

        testing = np.array([
            0.7142857142857143,
            0.7433414043583535,
            0.7602905569007264,
            0.7530266343825666,
            0.7554479418886199,
            0.7554479418886199,
            0.7627118644067796,
            0.7602905569007264,
            0.7602905569007264
        ])
        plt.plot(Cs, training, '-o', Cs, testing, '-o')
        # plt.legend(["Training", "Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")
        plt.title("Logistic Regression")

        ### Solver: Lbfgs
        Cs = [0.01, 0.03, 0.1, 0.3, 1, 1.3, 3, 6, 10]
        training = [
            0.8211036992116434,
            0.8835657974530018,
            0.9496664645239539,
            0.9909035779260158,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ]
        testing = [
            0.7142857142857143,
            0.7433414043583535,
            0.7602905569007264,
            0.7530266343825666,
            0.7554479418886199,
            0.7554479418886199,
            0.7627118644067796,
            0.7602905569007264,
            0.7602905569007264
        ]

        plt.plot(Cs, training, '-o', Cs, testing, '-o')
        # plt.title("T2")
        plt.legend(["N-cg Training", "N-cg Testing"] + ["Lbfgs Training", "Lbfgs Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")
        plt.show()

    # tunning
    elif True:
            ## Method: LogReg
            ### Solver: Newton-cg
            Cs = [0.08, 0.1, 0.12]
            training = np.array([
                0.9338993329290479,
                0.9496664645239539,
                0.9636143117040631,

            ])

            testing = np.array([
                0.7602905569007264,
                0.7602905569007264,
                0.7651331719128329,
            ])
            plt.plot(Cs, training, '-o', Cs, testing, '-o')
            # plt.legend(["Training", "Testing"])
            plt.xlabel("Inverse of Regularization Strength")
            plt.ylabel("Accuracy")

            ### Solver: Lbfgs
            Cs = [0.08, 0.1, 0.12]
            training = [
                0.9332929047907823,
                0.9496664645239539,
                0.9636143117040631,
            ]
            testing = [
                0.7602905569007264,
                0.7602905569007264,
                0.7651331719128329,
            ]

            plt.plot(Cs, training, '-o', Cs, testing, '-o')
            # plt.title("T1")
            plt.legend(["N-cg Training", "N-cg Testing"] + ["Lbfgs Training", "Lbfgs Testing"])
            plt.xlabel("Inverse of Regularization Strength")
            plt.ylabel("Accuracy")
            plt.title("Logistic Regression - Tuning")
            plt.show()

    # no regularization
    if True:
        ## Method: LogReg
        ### Solver: Newton-cg
        Cs = [10000000000, 100000000000, 1000000000000]
        training = np.array([
            1.0,
            1.0,
            1.0,

        ])

        testing = np.array([
            0.7360774818401937,
            0.7336561743341404,
            0.7360774818401937,
        ])
        plt.plot(Cs, training, '-o', Cs, testing, '-o')
        # plt.legend(["Training", "Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")

        ### Solver: Lbfgs
        Cs = [10000000000, 100000000000, 1000000000000]
        training = np.array([
            1.0,
            1.0,
            1.0,
        ])
        testing = [
            0.7506053268765133,
            0.7506053268765133,
            0.7506053268765133,
        ]

        plt.plot(Cs, training, '-o', Cs, testing, '-o')
        # plt.title("T1")
        plt.legend(["N-cg Training", "N-cg Testing"] + ["Lbfgs Training", "Lbfgs Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")
        plt.title("Logistic Regression - No Regularization")
        plt.show()
