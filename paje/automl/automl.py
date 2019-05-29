""" Automl Module
"""

from abc import ABC, abstractmethod

import numpy as np

from paje.base.component import Component
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics
from paje.module.modules import default_preprocessors, default_modelers


class AutoML(Component, ABC):
    def __init__(self, preprocessors=None, modelers=None,
                 storage_for_components=None, verbose=True,
                 random_state=0, storage=None,
                 show_warns=True, max_time=None, **kwargs):
        super().__init__(storage, show_warns, max_time, **kwargs)
        self.preprocessors = default_preprocessors \
            if preprocessors is None else preprocessors
        self.modelers = default_modelers if modelers is None else modelers
        if self.modelers is None:
            self.warning('No modelers given')
        self.random_state = random_state
        self.verbose = verbose
        self.storage_for_components = storage_for_components
        self.storage = None  # TODO: AutoML is only storing Pipelins for now

    def build_impl(self):
        # TODO: uncomment:
        # raise Exception('It is not clear if this class is instantiable yet.')
        pass

    def apply_impl(self, data):
        print('--------------------------------------------------------------')
        # print('max_iter', self.max_iter, '  max_depth', self.max_depth,
        #       '  static', self.static, '  fixed', self.fixed,
        #       '  repetitions', self.repetitions)
        evaluator = Evaluator(Metrics.error, "cv", 3,
                              random_state=self.random_state)

        failed, locked, tot = 0, 0, 0
        for i in range(self.max_iter):
            # Evaluates current hyperparameter (space-values) combination.
            pipelines = self.next_pipelines(data)

            errors = []
            for pipe in pipelines:
                tot += 1
                if self.verbose:
                    print(pipe)
                error = np.mean(evaluator.eval(pipe, data))
                if pipe.failed:
                    failed += 1
                if pipe.locked:
                    locked += 1
                errors.append(error)
            self.process(errors)
            if self.verbose:
                print("Current Error: ", error)
                print("Best Error: ", self.best_error, 'Locked/failed/total '
                                                       'pipelines:',
                      locked, '/', failed, '/', tot, '\n', )

        if self.verbose:
            print("Best pipeline found:")
            print(self.best())

        self.model = self.best()
        return self.model.apply(data)

    @abstractmethod
    def best(self):
        pass

    @abstractmethod
    def process(self, errors):
        pass

    def use_impl(self, data):
        return self.model.use(data)

    @abstractmethod
    def next_pipelines(self, data):
        pass

    # @abstractmethod
    # def next_dicts(self, forest):
    #     """
    #     This method defines the search heuristic and should be implemented by
    #     the child class.
    #     :return: a list of dictionaries or list of nested lists of dictionaries
    #     """
    #     pass

    @classmethod
    def tree_impl(cls, data=None):
        raise NotImplementedError("AutoML has no tree() implemented!")

    def fields_to_store_after_use(self):
        raise NotImplementedError(
            "AutoML has no fields_to_store_after_use() implemented!")

    def fields_to_keep_after_use(self):
        raise NotImplementedError(
            "AutoML has no fields_to_keep_after_use() implemented!")
