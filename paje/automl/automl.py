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
from paje.module.preprocessing.feature_selection.statistical_based.cfs import FilterCFS
from paje.module.preprocessing.feature_selection.statistical_based.chi_square import FilterChiSquare
from paje.module.preprocessing.scaler.equalization import Equalization
from paje.module.preprocessing.scaler.standard import Standard
from paje.pipeline.pipeline import Pipeline


class AutoML(Component, ABC):
    def init_impl(self, preprocessors=None, modelers=None,
                  method="all", max_iter=3, max_deepth=5, random_state=0):
        self.preprocessors = [FilterCFS, FilterChiSquare, RanOverSampler, RanUnderSampler,
                              Standard, Equalization] if preprocessors is None else preprocessors
        self.modelers = [RF, KNN, NB, DT, MLP, SVM, CB] if modelers is None else modelers
        self.random_state = random_state
        self.modules = [Equalization, RF]
        self.max_iter = max_iter

    def apply_impl(self, data):
        # Defines search space (space of hyperparameter spaces).
        self.forest = Pipeline(self.modules).hyperpar_spaces_forest(data)

        # Chooses the best hyperparameter space - hyperparameter values combination.
        best_error = 9999999
        for i in range(self.max_iter):
            pipe = Pipeline(self.modules, self.next_hyperpar_dicts())
            evaluator = Evaluator(data, Metrics.error, "cv", 3, self.random_state)
            error = np.mean(evaluator.eval(pipe, data))
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
        pass
