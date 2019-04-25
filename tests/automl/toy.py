from sys import argv

import sklearn.metrics

from paje.automl.random import RandomAutoML
from paje.automl.random_search_based import RadomSearchAutoML
from paje.data.data import Data

if len(argv) != 2:
    print('Usage: \npython toy.py dataset.arff')
else:
    data = Data.read_arff(argv[1], "class")
    X, y = data.data_x, data.data_y
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl_rs = RandomAutoML(fixed=False, repetitions=True, show_warnings=False,
                                  max_depth=4, max_iter=1, random_state=0)

    data_train = Data(X_train, y_train)
    data_test = Data(X_test, y_test)
    automl_rs.apply(data_train)
    resp = automl_rs.use(data_test).data_y

    # print(1 - np.sum(resp == data_test.data_y)/resp.shape[0])
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, resp))
