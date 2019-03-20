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
            np.warnings.filterwarnings('ignore')  # Supressing warnings due to NaN in linear algebra calculations.
            self.model.algorithm = 'brute'
            try:
                cov = np.cov(X)
                inv = np.linalg.pinv(cov)  # pinv is the same as inv for invertible matrices
                self.model.metric_params = {'VI': inv}
            except:
                # Uses a fake inverse of covariance matrix as fallback.
                self.model.metric_params = {'VI': np.eye(len(X))}
            np.warnings.filterwarnings('always')
        super().apply(data)

    def use(self, data):
        np.warnings.filterwarnings('ignore')  # Supressing warnings due to NaN in linear algebra calculations.
        res = super().use(data)
        np.warnings.filterwarnings('always')
        return res

    @staticmethod
    def hps_impl(data):
        kmax = floor(data.size() / 2)  # Assumes worst case of k-fold CV, i.e. k=2.
        dic = {
            'n_neighbors': ['z', 1, kmax],
            'metric': ['c', ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']],
            'weights': ['c', ['distance', 'uniform']]
        }
        return HPTree(dic, children=[])
