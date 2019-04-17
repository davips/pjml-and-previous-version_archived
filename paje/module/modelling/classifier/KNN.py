from math import *

import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier


class KNN(Classifier):
    def init_impl(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)

    def apply_impl(self, data):
        print('>>>>>>>>>>>>>>>>>>>>>>> ', data.n_instances())
        if self.model.metric == 'mahalanobis':
            X = data.data_x
            self.model.algorithm = 'brute'
            try:
                cov = np.cov(X)
                inv = np.linalg.pinv(cov)  # pinv is the same as inv for invertible matrices
                self.model.metric_params = {'VI': inv}
            except:
                # Uses a fake inverse of covariance matrix as fallback.
                self.model.metric_params = {'VI': np.eye(len(X))}
        return super().apply_impl(data)

    @classmethod
    def hps_impl(cls, data=None):
        print('> > > > > > > . > ', data.n_instances())
        cls.check_data(data)
        kmax = floor(data.n_instances() / 2 - 1)  # Assumes worst case of k-fold CV, i.e. k=2.
        print(kmax)
        dic = {
            'n_neighbors': ['z', kmax-1, kmax],
            'metric': ['c', ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']],
            'weights': ['c', ['distance', 'uniform']]
        }
        return HPTree(dic, children=[])
