from paje.data.data import Data
from paje.modelling.classifier.CB import CB
from paje.modelling.classifier.RF import RF
import sklearn.metrics
from sys import argv

if len(argv) != 2:
    print('Usage: \npython plot.py dataset.arff')
else:
    data = Data.read_arff(argv[1], "class")
    X, y = data.data_x, data.data_y
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    data_train = Data(X_train, y_train)
    data_test = Data(X_test, y_test)


    for trees in range(1, 100):
        model = CB(iterations=5*trees, verbose=False)
        # model = RF(n_estimators=5*trees)
        model.apply(data_train)
        resp = model.use(data_test)

        print(sklearn.metrics.accuracy_score(data_test.data_y, resp))
