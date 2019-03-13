from paje.automl.random_search_based import RadomSearchAutoML
from paje.base.data import Data
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

automl_rs = RadomSearchAutoML(fixed=False, repetitions=True, deep=5, max_iter=30)

data = Data.read_arff("../datasets/1012_flags.arff", "class")
sss = StratifiedShuffleSplit(n_splits=1, random_state=123)

for train_index, test_index in sss.split(data.data_x, data.data_y):
    X_train, X_test = data.data_x[train_index], data.data_x[test_index]
    y_train, y_test = data.data_y[train_index], data.data_y[test_index]
    automl_rs.apply(Data(data.data_x.copy(), data.data_y.copy()))
    resp = automl_rs.use(data)
    print(1 - np.sum(resp == data.data_y)/resp.shape[0])
