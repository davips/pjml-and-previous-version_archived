from numpy.random import choice
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, \
    StratifiedShuffleSplit

from paje.base.hps import ConfigSpace
from paje.ml.element.element import Element
from paje.util.encoders import json_unpack, json_pack


class CV(Element):
    def __init__(self, config, **kwargs):
        if 'iteration' not in config:
            config['iteration'] = 0
        super().__init__(config, **kwargs)
        self._memoized = {}
        self._max = 0
        # split, steps, test_size, random_state
        if self.config['split'] == "cv":
            self.model = StratifiedKFold(
                shuffle=True,
                n_splits=self.config['steps'],
                random_state=self.config['random_state'])
        elif self.config['split'] == "loo":
            self.model = LeaveOneOut()
        elif self.config['split'] == 'holdout':
            self.model = StratifiedShuffleSplit(
                n_splits=self.config['steps'],
                test_size=self.config['test_size'],
                random_state=self.config['random_state'])
        self.iteration = self.config['iteration']

    def next(self):
        if self.config['iteration'] == self._max:
            return None

        # Creates new, updated instance.
        newconfig = self.config.copy()
        newconfig['iteration'] += 1
        inst = CV(newconfig)
        inst._max = self._max
        inst._memoized = self._memoized
        return inst

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

    def tree_impl(self):
        config_space = ConfigSpace('CV')
        start = config_space.start()

        node = config_space.node()
        start.add_child(node)
        node.add_hp(CatHP('iteration', choice, a=[0]))

        holdout = config_space.node()
        node.add_child(holdout)
        holdout.add_hp(CatHP('split', choice, a=['holdout']))
        holdout.add_hp(IntHP('steps', choice, low=1, high=100000))
        holdout.add_hp(RealHP('test_size', choice, low=1e-06, high=1 - 1e-06))

        cv = config_space.node()
        node.add_child(cv)
        cv.add_hp(CatHP('split', choice, a=['cv']))
        cv.add_hp(IntHP('steps', choice, low=1, high=100000))

        loo = config_space.node()
        node.add_child(loo)
        loo.add_hp(CatHP('split', choice, a=['loo']))

        return config_space


# Needed classes to mark history of transformations when apply() give different
# results than use() for same input data.
class Apply:
    def __init__(self, component):
        self.component = component
        self.config = {'op': 'a',
                       'comp': json_unpack(self.component.serialized())}


class Use:
    def __init__(self, component):
        self.component = component
        self.config = {'op': 'u',
                       'comp': json_unpack(self.component.serialized())}
