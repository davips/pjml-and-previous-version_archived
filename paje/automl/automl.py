from abc import ABC, abstractmethod

import numpy as np

from paje.base.component import Component
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics
from paje.module.modelling.classifier.CB import CB
from paje.module.modelling.classifier.DT import DT
from paje.module.modelling.classifier.KNN import KNN
from paje.module.modelling.classifier.MLP import MLP
from paje.module.modelling.classifier.NB import NB
from paje.module.modelling.classifier.RF import RF
from paje.module.modelling.classifier.SVM import SVM
from paje.module.preprocessing.balancer.over.ran_over_sampler import RanOverSampler
from paje.module.preprocessing.balancer.under.ran_under_sampler import RanUnderSampler
from paje.module.preprocessing.data_reduction.DRPCA import DRPCA
from paje.module.preprocessing.feature_selection.statistical_based.cfs import FilterCFS
from paje.module.preprocessing.feature_selection.statistical_based.chi_square import FilterChiSquare
from paje.module.preprocessing.scaler.equalization import Equalization
from paje.module.preprocessing.scaler.standard import Standard
from paje.pipeline.pipeline import Pipeline

# TODO: Extract list of all modules automatically from the package module.
# PCA = Pipeline([Standard, DRPCA]) #, Pipeline([Standard, DRPCA]).hyperpar_spaces_forest(data))
default_preprocessors = [DRPCA, FilterCFS, FilterChiSquare, RanOverSampler,
                         RanUnderSampler, Standard, Equalization]
default_modelers = [RF, KNN, NB, DT, MLP, SVM, CB]


class AutoML(Component, ABC):
    def init_impl(self, preprocessors=None, modelers=None, repetitions=False,
                  method="all", max_iter=3, max_depth=5, fixed=True, random_state=0):
        self.random_state = random_state
        self.max_iter = max_iter
        self.fixed = fixed
        self.preprocessors = default_preprocessors if preprocessors is None else preprocessors
        self.modelers = default_modelers if modelers is None else modelers

    def apply_impl(self, data):
        best_error = 9999999
        for i in range(self.max_iter):
            # Defines search space (space of hyperparameter spaces).
            self.modules = [DRPCA, MLP]
            self.forest = Pipeline(self.modules).hyperpar_spaces_forest(data)

            # Evaluates current hyperparameterspace-hyperparametervalues combination.
            pipe = Pipeline(self.modules, self.next_hyperpar_dicts())
            evaluator = Evaluator(data, Metrics.error, "cv", 3, self.random_state)
            error = np.mean(evaluator.eval(pipe, data))
            print(pipe, '\nerror: ', error, '\n')
            if error < best_error:
                best_error = error
                self.model = pipe
        return self.model.apply(data)

    def use_impl(self, data):
        return self.model.use(data)

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        raise NotImplementedError("AutoML has neither hyper_spaces_tree() (obviously)",
                                  " nor hyper_spaces_forest() (not so obviously) implemented!")

    @abstractmethod
    def next_hyperpar_dicts(self):
        """
        This method defines the search heuristic and should be implemented by the child class.
        :return: a dictionary
        """
        pass
