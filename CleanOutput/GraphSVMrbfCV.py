import numpy as np
import matplotlib.pyplot as plt

## Method: SVM
### Solver: RBF
# TODO: dizer que tinhamos MAIS Cs que nao mostramos no plot pa nao puluir muito
# Cs = [0.1, 0.3, 1, 3, 6]
if False:    # without stratify
    print("no stratify")
    if True:    # 1st run
        gammas = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
        #### C = c: 0.1
        training = np.array([
            0.3959975742874469,
            0.6124924196482717,
            0.19648271679805943,
            0.10491206791995149,
            0.10491206791995149,
            0.10491206791995149,
        ])

        testing = np.array([
            0.29539951573849876,
            0.47699757869249393,
            0.1162227602905569,
            0.07990314769975787,
            0.07990314769975787,
            0.07990314769975787,
        ])
        plt.plot(gammas,training, '-o', gammas, testing, '-o')
        # plt.legend(["C = 0.1 Training", "C = 0.1 Testing"])
        plt.xlabel("Inverse of Regularization Strength")
        plt.ylabel("Accuracy")
        # plt.show()

        #### C = c: 1
        training = np.array([
            0.8696179502728927,
            0.9648271679805943,
            1.0,
            1.0,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.7312348668280871,
            0.784503631961259,
            0.7796610169491526,
            0.3099273607748184,
            0.07990314769975787,
            0.07990314769975787,
        ])
        plt.plot(gammas,training, '-o', gammas, testing, '-o')
        # plt.legend(["C = 0.1 Training", "C = 0.1 Testing"] + ["C = 1 Training", "C = 1 Testing"])
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        # plt.show()

        #### C = c:3
        training = np.array([
            0.9496664645239539,
            0.9993935718617344,
            1.0,
            1.0,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.7869249394673123,
            0.8208232445520581,
            0.8038740920096852,
            0.35108958837772397,
            0.07990314769975787,
            0.07990314769975787,
        ])
        plt.plot(gammas,training, '-o', gammas, testing, '-o')
        plt.legend(["C = 0.1 Training", "C = 0.1 Testing"] + ["C = 1 Training", "C = 1 Testing"] + ["C = 3 Training", "C = 3 Testing"])
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        plt.show()

    # tunning
    elif True:
        gammas = [0.008, 0.01, 0.012]
        #### C = c: 1
        training = np.array([
            0.9490600363856883,
            0.9648271679805943,
            0.9727107337780473,
        ])

        testing = np.array([
            0.784503631961259,
            0.784503631961259,
            0.7941888619854721,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')
        # plt.legend(["C = 1 Training", "C = 1 Testing"])
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        # plt.show()

        #### C = c: 2
        training = np.array([
            0.9860521528198909,
            0.9963614311704063,
            0.9987871437234688,
        ])

        testing = np.array([
            0.8087167070217918,
            0.8087167070217918,
            0.8159806295399515,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')
        # plt.legend(["C = 2 Training", "C = 2 Testing"])

        #### C = c: 3
        training = np.array([
            0.9981807155852032,
            0.9993935718617344,
            1.0,
        ])

        testing = np.array([
            0.8111380145278451,
            0.8208232445520581,
            0.8305084745762712,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')
        plt.legend(["C = 1 Training", "C = 1 Testing"] + ["C = 2 Training", "C = 2 Testing"] +["C = 3 Training", "C = 3 Testing"])
        plt.show()

else: # with stratify
    print("stratify")
    if True:    # 1st run
        gammas = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
        #### C = c: 0.1
        training = np.array([
            0.5318374772589448,
            0.7222559126743481,
            0.25955124317768347,
            0.20133414190418436,
            0.20133414190418436,
            0.20133414190418436,
        ])

        testing = np.array([
            0.4648910411622276,
            0.5907990314769975,
            0.2009685230024213,
            0.12106537530266344,
            0.1016949152542373,
            0.1016949152542373,
        ])
        plt.plot(gammas,training, '-o', gammas, testing, '-o')
        # plt.legend(["C = 0.1 Training", "C = 0.1 Testing"])
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        # plt.show()

        #### C = c: 1
        training = np.array([
            0.8623408126137053,
            0.9630078835657975,
            0.9993935718617344,
            1.0,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.7457627118644068,
            0.8135593220338984,
            0.7893462469733656,
            0.47699757869249393,
            0.10653753026634383,
            0.1016949152542373,
        ])
        plt.plot(gammas,training, '-o', gammas, testing, '-o')
        # plt.legend(["C = 0.1 Training", "C = 0.1 Testing"] + ["C = 1 Training", "C = 1 Testing"])
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        # plt.show()

        #### C = c:3
        training = np.array([
            0.9454214675560946,
            0.9993935718617344,
            1.0,
            1.0,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.8184019370460048,
            0.8595641646489104,
            0.8159806295399515,
            0.5084745762711864,
            0.10653753026634383,
            0.1016949152542373,
        ])
        plt.plot(gammas,training, '-o', gammas, testing, '-o')
        plt.legend(["C = 0.1 Training", "C = 0.1 Testing"] + ["C = 1 Training", "C = 1 Testing"] + ["C = 3 Training", "C = 3 Testing"])
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        plt.title("SVM - RBF Kernel")
        plt.show()

        # tunning
    if True:
        gammas = [0.008, 0.009, 0.01, 0.012, 0.014]
        #### C = c: 2
        training = np.array([
            0.9878714372346877,
            0.9927228623408126,
            0.9945421467556095,
            0.9951485748938751,
            0.9987871437234688
        ])

        testing = np.array([
            0.847457627118644,
            0.8450363196125908,
            0.8498789346246973,
            0.8595641646489104,
            0.8571428571428571
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        # plt.show()

        #### C = c: 2.5
        training = np.array([
            0.9927228623408126,
            0.9945421467556095,
            0.9957550030321407,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.8547215496368039,
            0.8547215496368039,
            0.8595641646489104,
            0.864406779661017,
            0.8595641646489104,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')

        #### C = c: 3
        training = np.array([
            0.9945421467556095,
            0.9969678593086719,
            0.9993935718617344,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.8547215496368039,
            0.8619854721549637,
            0.8595641646489104,
            0.864406779661017,
            0.8595641646489104,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')
        plt.legend(["C = 2 Training", "C = 2 Testing"] + ["C = 2.5 Training", "C = 2.5 Testing"] + ["C = 3 Training",
                                                                                                "C = 3 Testing"])
        plt.title("SVM - RBF Kernel - tuning")
        plt.show()

    # no regularization
    if True:
        gammas = [0.008, 0.01, 0.012]
        #### C = c: 1e10
        training = np.array([
            1.0,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.8547215496368039,
            0.8571428571428571,
            0.8692493946731235,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')
        plt.xlabel("Gammas")
        plt.ylabel("Accuracy")
        # plt.show()

        #### C = c: 1e11
        training = np.array([
            1.0,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.8547215496368039,
            0.8571428571428571,
            0.8692493946731235,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')

        #### C = c: 1e12
        training = np.array([
            1.0,
            1.0,
            1.0,
        ])

        testing = np.array([
            0.8547215496368039,
            0.8571428571428571,
            0.8692493946731235,
        ])
        plt.plot(gammas, training, '-o', gammas, testing, '-o')
        plt.legend(["C = 1e10 Training", "C = 1e10 Testing"] + ["C = 1e11 Training", "C = 1e11 Testing"] + ["C = 1e12 Training",
                                                                                                    "C = 1e12 Testing"])
        plt.title("SVM - RBF Kernel - No Regularization")
        plt.show()