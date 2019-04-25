from sys import argv

import sklearn.metrics

from paje.automl.random import RandomAutoML
from paje.data.data import Data
from paje.module.modelling.classifier.RF import RF
from paje.module.preprocessing.balancer.over.ran_over_sampler import RanOverSampler
from paje.module.preprocessing.balancer.under.ran_under_sampler import RanUnderSampler
from paje.module.preprocessing.data_reduction.DRPCA import DRPCA
from paje.module.preprocessing.feature_selection.statistical_based.cfs import FilterCFS
from paje.module.preprocessing.scaler.equalization import Equalization
from paje.module.preprocessing.scaler.standard import Standard

if len(argv) != 2:
    print('Usage: \npython toy.py dataset.arff')
else:
    data = Data.read_arff(argv[1], "class")
    X, y = data.data_x, data.data_y
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)
    data_train = Data(X_train, y_train)
    data_test = Data(X_test, y_test)

    automl_rs = RandomAutoML(preprocessors=[Equalization, DRPCA,
                                            RanOverSampler, Standard],
                             modelers=[RF], max_iter=2, static=True, fixed=True,
                             max_depth=50, repetitions=0, method="all")
    automl_rs.apply(data_train)
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    print()

    automl_rs = RandomAutoML(max_iter=2, static=False, fixed=True,
                             max_depth=4, repetitions=0, method="all")
    automl_rs.apply(data_train)
    resp = automl_rs.use(data_test).data_y
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    print()

    automl_rs = RandomAutoML(max_iter=2, static=False, fixed=True,
                             max_depth=10, repetitions=2, method="all")
    automl_rs.apply(data_train)
    resp = automl_rs.use(data_test).data_y
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    print()

    automl_rs = RandomAutoML(max_iter=2, static=False, fixed=False,
                             max_depth=8, repetitions=0, method="all")
    automl_rs.apply(data_train)
    resp = automl_rs.use(data_test).data_y
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    print()

    automl_rs = RandomAutoML(max_iter=2, static=False, fixed=False,
                             max_depth=10, repetitions=2, method="all")
    automl_rs.apply(data_train)
    resp = automl_rs.use(data_test).data_y
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    print()

    # print(1 - np.sum(resp == data_test.data_y)/resp.shape[0])
