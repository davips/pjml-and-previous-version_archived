""" Automl Module
"""

from abc import ABC, abstractmethod
import numpy as np
from paje.base.component import Component
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics
from paje.module.modelling.classifier.cb import CB
from paje.module.modelling.classifier.dt import DT
from paje.module.modelling.classifier.knn import KNN
from paje.module.modelling.classifier.mlp import MLP
from paje.module.modelling.classifier.nb import NB
from paje.module.modelling.classifier.rf import RF
from paje.module.modelling.classifier.svm import SVM
from paje.module.preprocessing.supervised.instance.balancer.over.\
        ran_over_sampler import RanOverSampler
from paje.module.preprocessing.supervised.instance.balancer.under.\
        ran_under_sampler import RanUnderSampler
from paje.module.preprocessing.unsupervised.feature.scaler.standard\
        import Standard
from paje.module.preprocessing.unsupervised.feature.transformer.drpca\
        import DRPCA
from paje.module.preprocessing.supervised.feature.selector.statistical.cfs\
        import FilterCFS
from paje.module.preprocessing.unsupervised.feature.scaler.equalization\
        import Equalization
from paje.pipeline.pipeline import Pipeline

# TODO: Extract list of all modules automatically from the package module.
# PCA = Pipeline([Standard, DRPCA]) #, Pipeline([Standard, DRPCA]).hyperpar_spaces_forest(data))
default_preprocessors = [DRPCA, FilterCFS, RanOverSampler,
                         RanUnderSampler, Standard, Equalization]
default_modelers = [RF, KNN, NB, DT, MLP, SVM, CB]


class AutoML(Component, ABC):

    def init_impl(self, preprocessors=None, modelers=None, random_state=0):
        self.random_state = random_state
        self.preprocessors = default_preprocessors \
            if preprocessors is None else preprocessors
        self.modelers = default_modelers if modelers is None else modelers
        if self.modelers is None:
            self.warning('No modelers given')

    def apply_impl(self, data):
        print('--------------------------------------------------------------')
        print('max_iter', self.max_iter, '  max_depth', self.max_depth,
              '  static', self.static, '  fixed', self.fixed,
              '  repetitions', self.repetitions)
        evaluator = Evaluator(data, Metrics.error, "cv", 3,
                              self.random_state)

        for i in range(self.max_iter):
            # Evaluates current hyperparameter (space-values) combination.
            pipelines = self.next_pipelines(data)

            errors = []
            for pipe in pipelines:
                # TODO:pq passamos data denovo ? OMG
                error = np.mean(evaluator.eval(pipe, data))
                errors.append(error)
                print(pipe, '\nerror: ', error, '\n')

            self.process(errors)

        self.model = self.best()
        return self.model.apply(data)

    @abstractmethod
    def best():
        pass

    @abstractmethod
    def process():
        pass

    def use_impl(self, data):
        return self.model.use(data)

    @abstractmethod
    def next_pipelines(self):
        pass

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        raise NotImplementedError("AutoML has neither hyper_spaces_tree()\
                                  (obviously)",
                                  " nor hyper_spaces_forest()\
                                  (not so obviously) implemented!")

    @abstractmethod
    def next_hyperpar_dicts(self, forest):
        """
        This method defines the search heuristic and should be implemented by the child class.
        :return: a list of dictionaries or list of nested lists of dictionaries
        """
        pass

    def handle_storage(self, data):
        # TODO: replicate this method to other nesting modules, not only Pipeline and AutoML
        return self.apply_impl(data)
