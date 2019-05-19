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
            try:
                if self.storage is not None:
                    # TODO: failed apply/use should store fake bad predictions
                    output_train = self.storage.get_or_else(
                        component, data_train, data_train,
                        component.apply
                    )

                    # Just in case the results are stored for apply(),
                    #  but not for use()...
                    def use(setin):
                        if component.unfit:
                            print("only apply()'s result was stored," +
                                  " not use()'s; completing process...")
                            # Unneeded to get output_train again.
                            component.apply(data_train)
                        return component.use(setin)

                    output_test = self.storage.get_or_else(
                        component, data_train, data_test,
                        use
                    )

                else:
                    output_train = component.apply(data_train)
                    output_test = component.use(data_test)

                error = self.metric(output_test)

            except ExceptionInApplyOrUse as e:
                # TODO: we are assuming that eval is minimizing an error measure
                error = 999666
                traceback.print_exc()
                print(e)
                exit(0)
            perfs.append(error)

        return perfs
