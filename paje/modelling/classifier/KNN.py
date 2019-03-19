from math import *

import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier

from paje.base.hps import HPTree


class KNN:
    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = None
        self.weights = weights

    def apply(self, data):
        X, y = data.xy()
        if self.metric == 'mahalanobis':
            np.warnings.filterwarnings('ignore')  # Supressing warnings due to NaN in linear algebra calculations.
            try:
                cov = np.cov(X)
                inv = np.linalg.pinv(cov)  # pinv is the same as inv for invertible matrices
                self.model = KNeighborsClassifier(algorithm='brute', metric_params={'VI': inv}, metric='mahalanobis', weights=self.weights, n_neighbors=self.n_neighbors).fit(X, y)
            except:
                # Uses a fake inverse of covariance matrix as fallback.
                self.model = KNeighborsClassifier(algorithm='brute', metric_params={'VI': np.eye(len(X))}, metric='mahalanobis', weights=self.weights, n_neighbors=self.n_neighbors).fit(X, y)
            np.warnings.filterwarnings('always')
        else:
            self.model = KNeighborsClassifier(weights=self.weights, n_neighbors=self.n_neighbors, metric=self.metric).fit(X, y)

    def use(self, data):
        X = data.data_x
        np.warnings.filterwarnings('ignore')  # Supressing warnings due to NaN in linear algebra calculations.
        res = self.model.predict(X)
        np.warnings.filterwarnings('always')
        return res

    def explain(self, X):
        raise NotImplementedError("Should it return a sorted list of neighbors?")

    @staticmethod
    def hps(data):
        kmax = floor(data.size() / 2)  # Assumes worst case of k-fold CV, i.e. k=2.
        data = {
            'n_neighbors': ['z', 1, kmax],
            'metric': ['c', ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']],
            'weights': ['c', ['distance', 'uniform']]
        }
        return HPTree(data, children=[])
