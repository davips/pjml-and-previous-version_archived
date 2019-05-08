from math import *
from catboost import CatBoostClassifier

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class CB(Classifier):
    def __init__(self, in_place=False, memoize=False,
                 show_warnings=True, verbose=False, **kwargs):
        self.verbose = verbose
        super().__init__(in_place, memoize, show_warnings, kwargs)

    # def apply(self, data):
    #     try:
    #         super().apply(data)
    #     except:
    #         if super().show_warnings:
    #             print('Falling back to random classifier, if there are convergence problems (bad "nu" value, for instance).')
    #         self.model = DummyClassifier(strategy='uniform').fit(*data.xy())

    def instantiate_model(self):
        self.model = CatBoostClassifier(**self.dict, verbose=self.verbose)

    @classmethod
    def tree_impl(cls, data=None):
        # todo: inconsistent pipelines: All features are either constant or ignored.
        cls.check_data(data)
        # todo: set random seed
        data_for_speed = {'iterations': ['z', [2, 1000]]}  # Entre outros
        n_estimators = min([500, floor(sqrt(data.n_instances() * data.n_attributes()))])

        dic = {
            'iterations': ['c', [n_estimators]],  # Only to set the default, not for search.
            'learning_rate': ['r', [0.000001, 1.0]],
            'depth': ['z', [1, 15]],  # Docs says 32, but CatBoostError says 16( but is 15? )
            'l2_leaf_reg': ['r', [0.01, 99999]],
            'loss_function': ['c', ['MultiClass']],
            'border_count': ['z', [1, 255]],
            # 'verbose': ['c', [False]],
            # 'ctr_border_count': ['z', [1,255]], # for categorical values
            # 'thread_count'
        }

        return HPTree(dic, children=[])
