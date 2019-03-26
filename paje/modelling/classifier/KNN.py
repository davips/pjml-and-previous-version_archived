from math import *

import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier

from paje.base.hps import HPTree
from paje.modelling.classifier.classifier import Classifier


class KNN(Classifier):
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)

    def apply(self, data):
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
        super().apply(data)

    @staticmethod
    def hps_impl(data=None):
        if data is None:
            print('KNN needs to know the size of the dataset to estimate the maximum allowed k.')
            exit(0)
        kmax = floor(data.n_instances() / 2)  # Assumes worst case of k-fold CV, i.e. k=2.
        dic = {
            'n_neighbors': ['z', 1, kmax],
            'metric': ['c', ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']],
            'weights': ['c', ['distance', 'uniform']]
        }
        return HPTree(dic, children=[])
