import traceback

from paje.base.exceptions import ExceptionInApplyOrUse
from paje.base.data import Data
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedShuffleSplit


class Evaluator():
    def __init__(self, metric, split="cv", steps=10, random_state=0,
                 storage=None):

        self.storage = storage
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

    # @profile
    def eval(self, component, data):
        perfs = []
        for train_index, test_index \
                in self.split.split(*data.xy):
            data_train = data.updated(X=data.X[train_index],
                                      y=data.y[train_index])
            data_test = data.updated(X=data.X[test_index], y=data.y[test_index])

            # TODO: failed pipeline store fake bad predictions,
            #  but only when self.storage is activated
            if self.storage is not None:
                output_train, output_test = self.storage.get_or_run(
                    component, data_train, data_test)
            else:
                # TODO: if output_train is needed, it should come from
                #  component.use(), not from component.apply()!
                component.apply(data_train)
                output_test = component.use(data_test)

            error = self.metric(output_test)
            perfs.append(error)

        return perfs
