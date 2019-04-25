from sys import argv

import sklearn.metrics

from paje.automl.random import RandomAutoML
from paje.data.data import Data
from paje.module.modelling.classifier.RF import RF
from paje.module.preprocessing.balancer.under.ran_under_sampler import RanUnderSampler
from paje.module.preprocessing.data_reduction.DRPCA import DRPCA
from paje.module.preprocessing.scaler.standard import Standard

if len(argv) != 2:
    print('Usage: \npython toy.py dataset.arff')
else:
    data = Data.read_arff(argv[1], "class")
    X, y = data.data_x, data.data_y
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl_rs = RandomAutoML(preprocessors=[DRPCA, RanUnderSampler, Standard], modelers=[RF],
                             max_iter=2, static=True,
                             fixed=True, max_depth=5,
                             repetitions=0, method="all",
                             random_state=0)

    data_train = Data(X_train, y_train)
    data_test = Data(X_test, y_test)
    automl_rs.apply(data_train)
    resp = automl_rs.use(data_test).data_y

    # print(1 - np.sum(resp == data_test.data_y)/resp.shape[0])
    print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, resp))
