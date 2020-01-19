import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from SuportVectorMachine.suport_vector_machine import SuportVectorMachine

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

iris = load_iris()
df_train = pd.DataFrame(iris.data, columns=iris.feature_names)
df_train['label'] = iris.target
# print(df_train)

data = np.array(df_train.iloc[:100, [0, 1, -1]])
for i in range(len(data)):
    if data[i, -1] == 0:
        data[i, -1] = -1

print(data)
X, y = data[:, :2], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2020)

plt.scatter(X[:50, 0], X[:50, 1], label='0')
plt.scatter(X[50:, 0], X[50:, 1], label='1')
plt.legend()
plt.show()

svm = SuportVectorMachine(max_iter=200)
svm.fit(X_train, y_train)

score = svm.score(X_test, y_test)
print(score)
