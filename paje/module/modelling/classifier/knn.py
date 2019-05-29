from math import *

import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier

from paje.base.exceptions import ExceptionInApplyOrUse
from paje.base.hps import HPTree
from paje.module.modelling.classifier.classifier import Classifier
from paje.util.distributions import exponential_integers


class KNN(Classifier):
    def build_impl(self):
        # Extract n_instances from hps to be available to be used in apply()
        # if neeeded.
        newdic = self.dic.copy()
        self.n_instances = newdic.get('@n_instances')
        if self.n_instances is not None:
            del newdic['@n_instances']
        self.model = KNeighborsClassifier(**newdic)

    def apply_impl(self, data):
        # If data underwent undersampling, rescale k as if its original interval of values was stretched to fit into the new size.
        # if data.n_instances < self.n_instances:
        #     pct = self.model.n_neighbors / self.n_instances
        #     self.model.n_neighbors = floor(pct * data.n_instances) + 1

        # TODO: decide how to handle this
        if self.model.n_neighbors > data.n_instances:
            raise ExceptionInApplyOrUse('excess of neighbors!')

        # # Handle complicated distance measures.
        if self.model.metric == 'mahalanobis':
            X = data.X
            self.model.algorithm = 'brute'
            # try:
            # pinv is the same as inv for invertible matrices
            # if self.model.metric == 'mahalanobis' and self.model.n_neighbors>500 \
            #         or data.n_instances>10000:

            if len(X) > 5000:
                raise ExceptionInApplyOrUse('Mahalanobis for too big data, '
                                            'matrix size:', len(X))
            cov = np.cov(X)
            inv = np.linalg.pinv(cov)
            self.model.metric_params = {'VI': inv}

            # except ExceptionInApplyOrUse as e:
            #     raise e
            # except Exception as e:
            #     self.warning('Problems with Mahalanobis, '
            #                  'falling back to identity! ' + e)
            #     # Uses a fake inverse of covariance matrix as fallback.
            #     self.model.metric_params = {'VI': np.eye(len(X))}

        return super().apply_impl(data)

    def isdeterministic(self):
        return True

    @classmethod
    def tree_impl(cls, data=None):
        # Assumes worst case of k-fold CV, i.e. k=2. Undersampling is another problem, handled by @n_instances.
        cls.check_data(data)
        kmax = min(1000, floor(data.n_instances / 2 - 1))

        dic = {
            'n_neighbors': ['c', exponential_integers(kmax)],
            'metric': ['c',
                       ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']],
            'weights': ['c', ['distance', 'uniform']],

            # Auxiliary - will used to calculate pct, only when the training set size happens to be smaller than kmax (probably due to under sampling).
            '@n_instances': ['c', [data.n_instances]]
        }
        return HPTree(dic, children=[])
