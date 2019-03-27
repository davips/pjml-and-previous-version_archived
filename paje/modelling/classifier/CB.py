from math import *
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier, Pool

from paje.base.hps import HPTree
from paje.modelling.classifier.classifier import Classifier


class CB(Classifier):
    def __init__(self, **kwargs):
        self.model = CatBoostClassifier(**kwargs)

    # def apply(self, data):
    #     try:
    #         super().apply(data)
    #     except:
    #         if super().show_warnings:
    #             print('Falling back to random classifier, if there are convergence problems (bad "nu" value, for instance).')
    #         self.model = DummyClassifier(strategy='uniform').fit(*data.xy())

    @classmethod
    def hps_impl(cls, data):
        cls.check_data(data)
        # todo: set random seed
        data_for_speed = {'iterations': ['z', 2, 1000]}  # Entre outros
        n_estimators = min([500, floor(sqrt(data.n_instances() * data.n_attributes()))])

        dic = {
            'iterations': ['c', [n_estimators]],  # Only to set the default, not for search.
            'learning_rate': ['r', 0.000001, 1.0],
            'depth': ['z', 1, 16], # Docs says 32, but CatBoostError says 16
            'l2_leaf_reg': ['r', 0.01, 99999],
            'loss_function': ['c', ['MultiClass']],
            'border_count': ['z', 1, 255],
            'verbose': ['c', [False]],
            # 'ctr_border_count': ['z', [1,255]], # for categorical values
            # 'thread_count'
        }

        return HPTree(dic, children=[])
