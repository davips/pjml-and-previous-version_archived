from math import *

from sklearn.ensemble import RandomForestClassifier

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class RF(Classifier):
    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)

    @classmethod
    def tree_impl(cls, data=None):
        cls.check_data(data)
        n_estimators = min([500, floor(sqrt(data.n_instances() * data.n_attributes()))])

        data_for_speed = {'n_estimators': ['z', [2, 1000]], 'max_depth': ['z', [2, data.n_instances()]]}  # Entre outros
        dic = {'bootstrap': ['c', [True, False]],
               'min_impurity_decrease': ['r', [0, 1]],
               'max_leaf_nodes': ['o', [2, 3, 5, 8, 12, 17, 23, 30, 38, 47, 57, 999999]],  # 999999 ~ None
               'max_features': ['r', [0.001, 1]],  # For some reason, the interval [1, n_attributes] didn't work for RF (maybe it is relative to a subset).
               'min_weight_fraction_leaf': ['r', [0, 0.5]],  # According to ValueError exception.
               'min_samples_leaf': ['z', [1, floor(data.n_instances() / 2)]],  # Int (# of instances) is better than float (proportion of instances) because different floats can collide to a same int, making intervals of useless real values.
               'min_samples_split': ['z', [2, floor(data.n_instances() / 2)]],  # Same reason as min_samples_leaf
               'max_depth': ['z', [2, data.n_instances()]],
               'criterion': ['c', ['gini', 'entropy']],  # Docs say that this parameter is tree-specific, but we cannot choose the tree.
               'n_estimators': ['c', [n_estimators]],  # Only to set the default, not for search.
               # See DT.py for more details about other settings.
               }
        return HPTree(dic, children=[])

    def instantiate_model(self):
        self.model = RandomForestClassifier(n_jobs=2, **self.dict)
