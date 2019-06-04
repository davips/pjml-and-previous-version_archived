from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


class Evaluator:
    def __init__(self, metric, split="cv", steps=10, random_state=0):

        self.metric = metric
        self.random_state = random_state
        self.steps = steps
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
        fold = 0
        for train_index, test_index in self.split.split(*data.Xy):
            name = f'{data.name()}_seed{self.random_state}' \
                f'_split{self.split.__class__.__name__}_fold'
            # print(train_index)
            data_train = data.updated(name=name + 'tr' + str(fold) +
                                           'steps' + str(self.steps) + 'split' +
                                           self.split.__class__.__name__,
                                      X=data.X[train_index],
                                      y=data.y[train_index])
            # print('uuid   tr ', data_train.uuid())
            data_test = data.updated(name=name + 'ts' + str(fold) +
                                          'steps' + str(self.steps) + 'split' +
                                          self.split.__class__.__name__,
                                     X=data.X[test_index],
                                     y=data.y[test_index])
            # print('uuid   tS ', data_test.uuid())

            # TODO: if output_train is needed, it should come from
            #  component.use(), not from component.apply()!
            component.apply(data_train)
            output_test = component.use(data_test)

            error = 1 if output_test is None else self.metric(output_test)
            perfs.append(error)
            fold += 1

        return perfs
