from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


class Evaluator:
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
        for train_index, test_index in self.split.split(*data.Xy):
            data_train = data.updated(X=data.X[train_index],
                                      y=data.y[train_index])
            data_test = data.updated(X=data.X[test_index], y=data.y[test_index])

            # TODO: if output_train is needed, it should come from
            #  component.use(), not from component.apply()!
            component.apply(data_train)
            output_test = component.use(data_test)

            error = 1 if output_test is None else self.metric(output_test)
            perfs.append(error)

        return perfs
