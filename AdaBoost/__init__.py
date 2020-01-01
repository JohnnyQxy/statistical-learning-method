from AdaBoost.adaboost import AdaBoost
import numpy as np

if __name__ == '__main__':
    X = np.arange(10).reshape(10, 1)
    y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])

    ada = AdaBoost(5, 0.5)
    ada.fit(X, y)
