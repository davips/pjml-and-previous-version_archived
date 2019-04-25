from abc import ABC, abstractmethod
import random

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
    def init_impl(self, preprocessors=None, modelers=None,
                  max_iter=2, static=True,
                  fixed=True, max_depth=5,
                  repetitions=0, method="all",
                  random_state=0):
        """
        AutoML
        :param preprocessors: list of modules for balancing, noise removal, sampling etc.
        :param modelers: list of modules for prediction (classification or regression etc.)
        :param repetitions: how many times can a module appear in a pipeline
        :param method: TODO
        :param max_iter: maximum number of pipelines to evaluate
        :param max_depth: maximum length of a pipeline
        :param static: are the pipelines generated always exactly as given by the ordered list preprocessors + modelers?
        :param fixed: are the pipelines generated always with length max(max_depth, len(preprocessors + modelers))?
        :param random_state: TODO
        :return:
        """
        if static and not fixed:
            self.error('static and not fixed!')
        if static and repetitions > 0:
            self.error('static and repetitions > 0!')
        self.random_state = random_state
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.static = static
        self.fixed = fixed
        self.repetitions = repetitions
        self.preprocessors = default_preprocessors \
            if preprocessors is None else preprocessors
        self.modelers = default_modelers if modelers is None else modelers
        if len(self.modelers) is 0:
            self.warning('No modelers given')
        if static:
            if len(self.modelers) > 1:
                self.warning('Multiple modelers given in static mode.')
            self.static_pipeline = self.preprocessors + self.modelers

    @abstractmethod
    def choose_modules(self):
        pass

    def apply_impl(self, data):
        best_error = 9999999
        for i in range(self.max_iter):
            # Defines search space (space of hyperparameter spaces).
            modules = self.static_pipeline if self.static else self.choose_modules()
            forest = Pipeline(modules).hyperpar_spaces_forest(data)

            # Evaluates current hyperparameter (space-values) combination.
            pipe = Pipeline(modules, self.next_hyperpar_dicts(forest))
            # pipe = Pipeline([Pipeline(
            #     [Standard, DRPCA],
            #     [{'@with_mean/std': (True, False)}, {'n_components': 2}]
            # )])
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
    def next_hyperpar_dicts(self, forest):
        """
        This method defines the search heuristic and should be implemented by the child class.
        :return: a list of dictionaries or list of nested lists of dictionaries
        """
        pass
