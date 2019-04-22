from math import *

import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier

from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier
from paje.util.distributions import exponential_integers


class KNN(Classifier):
    def init_impl(self, **kwargs):
        # Extract n_instances from hps to be available to be used in apply() if neeeded.
        self.n_instances = kwargs.get('@n_instances')
        del kwargs['@n_instances']
        self.model = KNeighborsClassifier(**kwargs)

    def apply_impl(self, data):
        # If data underwent undersampling, rescale k as if its original interval of values was stretched to fit into the new size.
        if data.n_instances() < self.n_instances:
            pct = self.model.n_neighbors / self.n_instances
            self.model.n_neighbors = floor(pct * data.n_instances()) + 1

        # Handle complicated distance measures.
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
    def hyperpar_spaces_tree_impl(cls, data=None):
        # Assumes worst case of k-fold CV, i.e. k=2. Undersampling is another problem, handled by @n_instances.
        cls.check_data(data)
        kmax = floor(data.n_instances() / 2 - 1)

        dic = {
            'n_neighbors': ['c', exponential_integers(kmax)],
            'metric': ['c', ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']],
            'weights': ['c', ['distance', 'uniform']],

            # Auxiliary - will used to calculate pct, only when the training set size happens to be smaller than kmax (probably due to under sampling).
            '@n_instances': ['c', [data.n_instances()]]
        }
        return HPTree(dic, children=[])
