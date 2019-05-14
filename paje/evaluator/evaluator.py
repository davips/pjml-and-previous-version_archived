from sklearn.model_selection import StratifiedShuffleSplit

from paje.base.exceptions import ExceptionInApply
from paje.data.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedShuffleSplit


class Evaluator():
    def __init__(self, metric, split="cv", steps=10, random_state=0):

        self.metric = metric
        if split == "cv":
            self.split = StratifiedKFold(n_splits=steps,
                                         shuffle=True,
                                         random_state=random_state)
        elif split == "loocv":
            self.split = LeaveOneOut()
        elif split == "holdout":
            self.split = StratifiedShuffleSplit(n_splits=steps, test_size=0.30,
                                                random_state=random_state)

    def eval(self, component, data):
        perfs = []
        for train_index, test_index \
                in self.split.split(data.data_x, data.data_y):
            data_train = Data(data.data_x[train_index],
                              data.data_y[train_index])
            data_test = Data(data.data_x[test_index], data.data_y[test_index])
            try:
                output_train = component.apply(data_train).data_y
                output_test = component.use(data_test).data_y
                error = self.metric(data_test, output_test)
            except ExceptionInApply as e:
                # TODO: we are assuming that eval is minimizing an error measure
                error = 999666
                print(e)
                # raise e
            perfs.append(error)

        return perfs
