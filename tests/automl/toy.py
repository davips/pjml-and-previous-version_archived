from sys import argv

import sklearn.metrics
from sklearn.decomposition import PCA

from paje.automl.random import RandomAutoML
from paje.base.data import Data
from paje.evaluator.metrics import Metrics
from paje.module.modules import default_preprocessors, default_modelers

# @profile
from paje.module.preprocessing.unsupervised.feature.transformer.drpca import \
    DRPCA
from paje.result.mysql import MySQL
from paje.result.sqlite import SQLite


def main():
    if len(argv) < 2 or len(argv) > 5:
        print('Usage: \npython toy.py dataset.arff '
              '[memoize? True/False] [iterations] [seed]')
    else:
        # storage = SQLite(debug=not True) if len(argv) >= 3 and argv[2] == \
        #                                     'True' else None
        storage = None #MySQL(db='teste', debug=not True) \
            # if len(argv) >= 3 and argv[2] == 'True' else None

        iterations = 30 if len(argv) < 4 else int(argv[3])
        random_state = 0 if len(argv) < 5 else int(argv[4])
        data = Data.read_arff(argv[1], "class")
        for a in argv:
            print(a)
        print('seed=', random_state)
        trainset, testset = data.split()

        # SQLite().setup()
        automl_rs = RandomAutoML(storage_for_components=storage,
                                 preprocessors=[DRPCA()],
                                 modelers=default_modelers, max_iter=iterations,
                                 static=False, fixed=False,
                                 max_depth=15, repetitions=0, method="all",
                                 show_warns=False,
                                 random_state=random_state).build()
        automl_rs.apply(trainset)
        testout = automl_rs.use(testset)
        print("Accuracy score", Metrics.error(testout))
        print()


if __name__ == '__main__':
    main()

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
