from math import *

from sklearn.ensemble import RandomForestClassifier

from paje.base.hps import HPTree
from paje.modelling.classifier.classifier import Classifier


class RandomForest(Classifier):
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    # def __init__(self, n_estimators=200, bootstrap=True, min_impurity_decrease=0, max_leaf_nodes=None, max_features='auto', min_weight_fraction_leaf=0, min_samples_leaf=1):
    #     self.model = RandomForestClassifier(n_estimators, bootstrap, min_impurity_decrease, max_leaf_nodes, max_features, min_weight_fraction_leaf, min_samples_leaf, max_depth=2)

    @staticmethod
    def hps_impl(data=None):
        if data is None:
            print('RF needs dataset to be able to estimate maximum values for some hyperparameters.')
            exit(0)
        n_estimators = min([500, floor(sqrt(data.n_instances() * data.n_attributes()))])

        data_for_speed = {'n_estimators': ['z', 2, 1000], 'max_depth': ['z', 2, data.n_instances()]} # Entre outros
        dic = {'bootstrap': ['c', [True, False]],
                'min_impurity_decrease': ['r', 0, 1],
                'max_leaf_nodes': ['o', [2, 4, 7, 11, 16, 22, 29, 37, 46, 999999]],  # 999999 ~ None
                'max_features': ['c', [None, 'sqrt', 'log2']], # It seems from the docs that 'int' == None and 'sqrt' == 'auto'. 'int' didn't work.
                'min_weight_fraction_leaf': ['r', 0, 0.5], # According to ValueError exception.
                'min_samples_leaf': ['z', 1, floor(data.n_instances() / 2)], # Int (# of instances) is better than float (proportion of instances) because different floats can collide to a same int, making intervals of useless real values.
                'min_samples_split': ['z', 2, floor(data.n_instances() / 2)], # Same reason as min_samples_leaf
                'max_depth': ['z', 2, data.n_instances()],
                'criterion': ['c', ['gini', 'entropy']], # Docs say that this parameter is tree-specific, but we cannot choose the tree.
                'n_estimators': ['c', [n_estimators]] # Only to set the default, not for search.
                }
        return HPTree(dic, children=[])
