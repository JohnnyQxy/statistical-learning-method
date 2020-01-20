import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from Perceptron.perceptron import Perceptron

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)

iris = load_iris()
df_train = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(df_train)
df_train['label'] = iris.target
data = np.array(df_train.iloc[:100, [0, 1, -1]])
print(data)
X, y = data[:, [0, 1]], data[:, -1]
y = np.array([1 if i == 1 else -1 for i in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)

perceptron = Perceptron(max_iter=1000)
print(perceptron.fit(X_train, y_train))

dotx = np.linspace(4, 7, 10)
doty = -(perceptron.coef_[0] * dotx + perceptron.intercept_) / perceptron.coef_[1]
plt.plot(dotx, doty)
plt.plot(X[:50, 0], X[:50, 1], 'bo', color='blue', label='0')
plt.plot(X[50:100, 0], X[50:100, 1], 'bo', color='orange', label='1')
plt.legend()
plt.show()
print(perceptron.coef_)
