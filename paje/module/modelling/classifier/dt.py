from math import *

from sklearn.tree import DecisionTreeClassifier

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class DT(Classifier):
    def build_impl(self):
        self.model = DecisionTreeClassifier(**self.dic)

    @classmethod
    def tree_impl(cls, data=None):
        # todo: set random seed
        cls.check_data(data)
        dic = {
            'criterion': ['c', ['gini', 'entropy']],
            'splitter': ['c', ['best', 'random']],
            'max_depth': ['z', [2, data.n_instances()]],
            'min_samples_split': ['z', [2, floor(data.n_instances() / 2)]],
            # Same reason as min_samples_leaf
            'min_samples_leaf': ['z', [1, floor(data.n_instances() / 2)]],
            # Int (# of instances) is better than float (proportion of instances) because different floats can collide to a same int, making intervals of useless real values.
            'min_weight_fraction_leaf': ['r', [0, 0.5]],
            # According to ValueError exception (tested only with RF).
            'max_features': ['r', [0.001, 1]],
            # For some reason, the interval [1, n_attributes] didn't work for DT.
            'max_leaf_nodes': ['o',
                               [2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 999999]],
            # 999999 ~ None
            'min_impurity_decrease': ['r', [0, 1]]
            # 'min_impurity_split': None, # deprecated in favour of min_impurity_decrease
            # 'class_weight': None, # We assume classes have equal weights.
            # 'presort': False # Strange setting that slow down large datasets and speed up small ones.
        }
        return HPTree(dic, children=[])
