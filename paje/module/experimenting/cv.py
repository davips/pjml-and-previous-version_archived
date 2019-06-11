from paje.base.component import Component
from paje.base.hps import HPTree
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, \
    StratifiedShuffleSplit


class CV(Component):
    def fields_to_keep_after_use(self):
        return 'all'

    def fields_to_store_after_use(self):
        return 'all'

    def build_impl(self):
        # split, steps, test_size, random_state
        if self.dic['split'] == "cv":
            self.model = StratifiedKFold(
                shuffle=True,
                n_splits=self.dic['steps'],
                random_state=self.dic['random_state'])
        elif self.dic['split'] == "loo":
            self.model = LeaveOneOut()
        elif self.dic['split'] == 'holdout':
            self.model = StratifiedShuffleSplit(
                n_splits=self.dic['steps'],
                test_size=self.dic['test_size'],
                random_state=self.dic['random_state'])
        self.testing_fold = self.dic['testing_fold']

    def apply_impl(self, data):
        indices, _ = list(self.model.split(*data.Xy))[self.testing_fold]
        self._applied_data_uuid = data.uuid()
        return data.updated(self, X=data.X[indices], Y=data.Y[indices])

    def use_impl(self, data):
        if self._applied_data_uuid != data.uuid():
            raise Exception('apply() and use() must partition the same data!')
        _, indices = list(self.model.split(*data.Xy))[self.testing_fold]
        return data.updated(self, X=data.X[indices], Y=data.Y[indices])

    def tree_impl(self, data):
        holdout = {
            'split': ['c', ['holdout']],
            'steps': ['z', [1, 100000]],  # TODO: which interval is good?
            'test_size': ['r', [0.000001, 0.999999]]
        }
        cv = {
            'split': ['c', ['cv']],
            'steps': ['z', [1, 100000]],  # TODO: which intervals are good?
        }
        loo = {
            'split': ['c', ['loo']],
        }
        HPTree({'testing_fold': ['z', [0, 100000]]}, [holdout, cv, loo])
