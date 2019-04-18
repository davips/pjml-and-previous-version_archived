from sklearn.externals import joblib

from paje.data.data import Data
from paje.module.modelling.classifier.RF import RF
import sklearn.metrics
from sys import argv

# Some tests to evaluate the resulting size of model dumps.
test_size = 1000


def f(n):
    return int(round(n * 1000))


if len(argv) != 4:
    print('Usage: \npython dump.py n_attributes n_classes n_instances')
else:
    data = Data.random(int(argv[1]), int(argv[2]), int(argv[3]))
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data.data_x, data.data_y, random_state=1, test_size=int(int(argv[3]) / 2))

    data_test = Data(x_test, y_test)
    data_train = Data(x_train, y_train)

    import numpy as np
    import sklearn

    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    np.random.seed(1234)

    for x0 in range(1, 100):
        x = int(round((pow(x0, 2))))
        model = RF(n_estimators=x)
        model.show_warnings = False
        tr = model.apply(data_train).data_y
        ts = model.use(data_test).data_y
        joblib.dump(model.model, model.__class__.__name__ +  str(x) + '.dump', compress=('bz2', 9))

        print(str(x) + "\t" + str(f(sklearn.metrics.accuracy_score(data_train.data_y, tr))) + "\t" + str(f(sklearn.metrics.accuracy_score(data_test.data_y, ts))))
