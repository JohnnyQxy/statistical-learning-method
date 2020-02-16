import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from NativeBayes.native_bayes import NativeBayes
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)

if __name__ == '__main__':
    iris = load_iris()
    df_data = pd.DataFrame(iris.data, columns=iris.feature_names)

    df_data['label'] = iris.target

    data = np.array(df_data.iloc[:100, :])
    X, y = data[:, :2], data[:, -1]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

    nb = NativeBayes()
    nb.fit(train_X, train_y)

    print(nb.score(test_X, test_y))
