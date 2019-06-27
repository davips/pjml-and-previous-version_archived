from math import *

from sklearn.tree import DecisionTreeClassifier

from paje.base.hps import HPTree
from paje.component.element.modelling.classifier.classifier import Classifier


class DT(Classifier):
    def build_impl(self):
        self.model = DecisionTreeClassifier(**self.dic)

    @classmethod
    def tree_impl(self):
        # todo: set random seed
        dic = {
            'criterion': ['c', ['gini', 'entropy']],
            'splitter': ['c', ['best']],
            'max_depth': ['z', [2, 1000]],
            'min_samples_split': ['r', [1e-6, 0.3]],
            # Same reason as min_samples_leaf
            'min_samples_leaf': ['r', [1e-6, 0.3]],
            # Int (# of instances) is better than float
            # (proportion of instances) because different floats can collide to
            # a same int, making intervals of useless real values.
            'min_weight_fraction_leaf': ['r', [0, 0.3]],
            'max_features': ['c', ['auto', 'sqrt', 'log2', None]],
            # For some reason, the interval [1, n_attributes] didn't work for
            # DT.
            # 'max_leaf_nodes': ['o',
            #                    [2, 4, 7, 11, 16, 22, 29, 37, 46, 56, None]],
            'min_impurity_decrease': ['r', [0.0, 0.2]],
            # 'min_impurity_split': None, # deprecated in favour
            # of min_impurity_decrease
            'class_weight': ['c', [None, 'balanced']],
            # 'presort': False # Strange setting that slow down large datasets
            # and speed up small ones.
        }
        return HPTree(dic, children=[])
