from sklearn.model_selection import StratifiedKFold, LeaveOneOut, \
    StratifiedShuffleSplit

from paje.base.hps import HPTree
from paje.ml.element.element import Element
from paje.util.encoders import json_unpack, json_pack
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, \
    StratifiedShuffleSplit

from paje.base.hps import HPTree
from paje.ml.element.element import Element
from paje.util.encoders import json_unpack, json_pack


class CV(Element):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memoized = {}
        self._max = 0

    def next(self):
        if self.args_set['iteration'] == self._max:
            return None

        # Creates new, updated instance.
        newdic = self.args_set.copy()
        newdic['iteration'] += 1
        inst = self.build(**newdic)
        inst._max = self._max
        inst._memoized = self._memoized
        return inst

    def build(self, iteration=0, **kwargs):
        if iteration == 0:
            self._memoized = {}
        kwargs['iteration'] = iteration
        return super().build(**kwargs)

    def build_impl(self, **args_set):
        # split, steps, test_size, random_state
        if self.args_set['split'] == "cv":
            self.model = StratifiedKFold(
                shuffle=True,
                n_splits=self.args_set['steps'],
                random_state=self.args_set['random_state'])
        elif self.args_set['split'] == "loo":
            self.model = LeaveOneOut()
        elif self.args_set['split'] == 'holdout':
            self.model = StratifiedShuffleSplit(
                n_splits=self.args_set['steps'],
                test_size=self.args_set['test_size'],
                random_state=self.args_set['random_state'])
        self.iteration = self.args_set['iteration']

    def apply_impl(self, data):
        if data.uuid() in self._memoized:
            partitions = self._memoized[data.uuid()]
        else:
            partitions = self._memoized[data.uuid()] = \
                list(self.model.split(*data.Xy))
        self._max = len(partitions) - 1
        indices, _ = partitions[self.iteration]

        # for sanity check in use()
        self._applied_data_uuid = data.uuid()

        return data.updated(Apply(self), X=data.X[indices], Y=data.Y[indices])

    def use_impl(self, data):
        # sanity check
        if self._applied_data_uuid != data.uuid():
            raise Exception('apply() and use() must partition the same data!')

        _, indices = list(self._memoized[data.uuid()])[self.iteration]

        return data.updated(Use(self), X=data.X[indices], Y=data.Y[indices])

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
        HPTree({'iteration': ['c', [0]]}, [holdout, cv, loo])


# Needed classes to mark history of transformations when apply() give different
# results than use().
class Apply:
    def __init__(self, component):
        self.component = component

    def serialized(self):
        return json_pack(
            {'op': 'a', 'comp': json_unpack(self.component.serialized())}
        )


class Use:
    def __init__(self, component):
        self.component = component

    def serialized(self):
        return json_pack(
            {'op': 'u', 'comp': json_unpack(self.component.serialized())}
        )
