from sys import argv

import sklearn.metrics

from paje.automl.random import RandomAutoML
from paje.data.data import Data
from paje.module.modules import default_preprocessors, default_modelers

if len(argv) < 2 or len(argv) > 5:
    print('Usage: \npython toy.py dataset.arff '
          '[memoize? True/False] [iterations] [seed]')
else:
    memoize = False if len(argv) < 3 else bool(argv[2])
    iterations = 30 if len(argv) < 4 else int(argv[3])
    random_state = 0 if len(argv) < 5 else int(argv[4])
    data = Data.read_arff(argv[1], "class")
    X, y = data.data_x, data.data_y
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    data_train = Data(X_train, y_train)
    data_test = Data(X_test, y_test)

    automl_rs = RandomAutoML(memoize=memoize,
                             preprocessors=default_preprocessors,
                             modelers=default_modelers, max_iter=iterations,
                             static=False, fixed=False,
                             max_depth=3, repetitions=1, method="all",
                             show_warns=False, random_state=random_state)
    automl_rs.apply(data_train)
    print("Accuracy score",
          sklearn.metrics.accuracy_score(data_test.data_y,
                                         automl_rs.use(data_test).data_y))
    print()

    # automl_rs = RandomAutoML(max_iter=n, static=False, fixed=True,
    #                          max_depth=4, repetitions=0, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()
    #
    # automl_rs = RandomAutoML(max_iter=n, static=False, fixed=True,
    #                          max_depth=10, repetitions=2, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()
    #
    # automl_rs = RandomAutoML(max_iter=n, static=False, fixed=False,
    #                          max_depth=8, repetitions=0, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()
    #
    # automl_rs = RandomAutoML(max_iter=2, static=False, fixed=False,
    #                          max_depth=10, repetitions=2, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()

    # print(1 - np.sum(resp == data_test.data_y)/resp.shape[0])
