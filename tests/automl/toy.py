from paje.automl.random_search_based import RadomSearchAutoML
from sklearn.model_selection import StratifiedShuffleSplit
from paje.data.data import Data
import numpy as np
import sklearn.model_selection
import sklearn.metrics
from sys import argv

if len(argv) != 2:
    print('Usage: \npython toy.py dataset.arff')
else:
    data = Data.read_arff(argv[1], "class")
    X, y = data.data_x, data.data_y
    X_train, X_test, y_train, y_test = \
                    sklearn.model_selection.train_test_split(X, y, random_state=1)


    automl_rs = RadomSearchAutoML(fixed=False, repetitions=True,
                                  deep=5, max_iter=10, random_state=0)

    data_train = Data(X_train, y_train)
    data_test = Data(X_test, y_test)
    automl_rs.apply(data_train)
    resp = automl_rs.use(data_test)

    # print(1 - np.sum(resp == data_test.data_y)/resp.shape[0])
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, resp))
