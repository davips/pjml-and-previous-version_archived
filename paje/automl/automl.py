""" Automl Module
"""

from abc import ABC, abstractmethod

import numpy as np

from paje.base.component import Component
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics
from paje.module.modules import default_preprocessors, default_modelers


class AutoML(Component, ABC):

    def __init__(self, preprocessors=None, modelers=None, random_state=0,
                 in_place=False, memoize=False, show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, **kwargs)
        self.preprocessors = default_preprocessors \
            if preprocessors is None else preprocessors
        self.modelers = default_modelers if modelers is None else modelers
        if self.modelers is None:
            self.warning('No modelers given')
        self.random_state = random_state

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
                try:
                    error = np.mean(evaluator.eval(pipe, data))
                except Exception as e:
                    error = None
                    print(e)
                    print()
                errors.append(error)
                print(pipe, '\nerror: ', error, '\n')

            self.process(errors)

        self.model = self.best()
        return self.model.apply(data)

    @abstractmethod
    def best(self):
        pass

    @abstractmethod
    def process(self):
        pass

    def use_impl(self, data):
        return self.model.use(data)

    @abstractmethod
    def next_pipelines(self, data):
        pass

    @abstractmethod
    def next_hyperpar_dicts(self, forest):
        """
        This method defines the search heuristic and should be implemented by
        the child class.
        :return: a list of dictionaries or list of nested lists of dictionaries
        """
        pass

    def handle_storage(self, data):
        # TODO: replicate this method to other nesting modules, not only Pipeline and AutoML
        return self.apply_impl(data)

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        raise NotImplementedError("AutoML has neither hyper_spaces_tree()\
                                  (obviously)",
                                  " nor hyper_spaces_forest()\
                                  (not so obviously) implemented!")
