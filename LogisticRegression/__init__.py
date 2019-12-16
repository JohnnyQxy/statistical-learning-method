import numpy as np
import matplotlib.pyplot as plt

from LogisticRegression.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split

data = []
label = []

f = open('data/testSet.txt')
for line in f.readlines():
    line = line.strip().split()
    data.append([float(line[0]), float(line[1])])
    label.append(int(line[-1]))

data = np.array(data)
label = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=20191216)

LR = LogisticRegression()
LR.fit_gd(X_train, y_train)
y_predict = LR.predict(X_test)
score = LR.score(X_test, y_test)
print('the result is ', score)


def plot_function(data, label, weights=None):
    n = np.shape(data)[0]
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=100, c='red', marker='s')
    ax.scatter(x2, y2, s=100, c='black')
    if weights is not None:
        x = np.arange(-3.0, 3.0, 0.1)
        y = -(weights[0] + weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()


plot_function(data, label, LR._theta)
