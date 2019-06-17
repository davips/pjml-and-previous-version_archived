from paje.base.component import Component
from paje.base.hps import HPTree
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, \
    StratifiedShuffleSplit


class CV(Component):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memoized = {}
        self._max = 0

    # def still_compatible_fields(self):
    #     return ''
    #
    # def touched_fields(self):
    #     return 'all'

    def next(self):
        if self.dic['testing_fold'] == self._max:
            return None

        # Creates new, updated instance.
        newdic = self.dic.copy()
        newdic['testing_fold'] += 1
        inst = self.build(**newdic)
        inst._max = self._max
        inst._memoized = self._memoized
        return inst

    def build(self, testing_fold=0, **kwargs):
        if testing_fold == 0:
            self._memoized = {}
        kwargs['testing_fold'] = testing_fold
        return super().build(**kwargs)

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
        if data.uuid() in self._memoized:
            partitions = self._memoized[data.uuid()]
        else:
            partitions = self._memoized[data.uuid()] = \
                list(self.model.split(*data.Xy))
        self._max = len(partitions) - 1
        indices, _ = partitions[self.testing_fold]

        # for sanity check in use()
        self._applied_data_uuid = data.uuid()

        return data.updated(self, X=data.X[indices], Y=data.Y[indices])

    def use_impl(self, data):
        # sanity check
        if self._applied_data_uuid != data.uuid():
            raise Exception('apply() and use() must partition the same data!')

        _, indices = list(self._memoized[data.uuid()])[self.testing_fold]

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
        HPTree({'testing_fold': ['c', [0]]}, [holdout, cv, loo])
