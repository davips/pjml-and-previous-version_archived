""" TODO the docstring documentation
"""
# from python
from abc import ABC, abstractmethod

# from other packages
import numpy as np

# from paje
from paje.base.component import Component
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics
from paje.module.modules import default_preprocessors, default_modelers


class AutoML(Component, ABC):
    """ TODO the docstring documentation
    """
    def __init__(self, preprocessors=None, modelers=None,
                 storage_for_components=None, verbose=True,
                 random_state=0, storage=None,
                 show_warns=True, max_time=None, **kwargs):
        super().__init__(storage, show_warns, max_time, **kwargs)
        """ TODO the docstring documentation
        """
        self.preprocessors = default_preprocessors \
            if preprocessors is None else preprocessors
        self.modelers = default_modelers if modelers is None else modelers
        if self.modelers is None:
            self.warning('No modelers given')
        self.random_state = random_state
        self.verbose = verbose
        self.storage_for_components = storage_for_components
        self.storage = None  # TODO: AutoML is only storing Pipelins for now

    def apply_impl(self, data):
        """ TODO the docstring documentation
        """
        print('--------------------------------------------------------------')
        # print('max_iter', self.max_iter, '  max_depth', self.max_depth,
        #       '  static', self.static, '  fixed', self.fixed,
        #       '  repetitions', self.repetitions)
        evaluator = Evaluator(Metrics.error, "cv", 10,
                              random_state=self.random_state)

        failed, locked, succ, tot = 0, 0, 0, 0
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
                elif pipe.locked:
                    locked += 1
                else:
                    succ += 1
                errors.append(error)
            self.process(errors)
            if self.verbose:
                print("Current Error: ", error)
                print("Best Error: ", self.best_error, 'Locks/fails/successes:',
                      locked, '/', failed, '/', succ, '\n', )

        if self.verbose:
            print("Best pipeline found:")
            print(self.best())

        self.model = self.best()
        return self.model.apply(data)

    def use_impl(self, data):
        return self.model.use(data)

    def build_impl(self):
        # TODO: uncomment:
        # raise Exception('It is not clear if this class is instantiable yet.')
        pass

    @abstractmethod
    def best(self):
        pass

    @abstractmethod
    def process(self, errors):
        pass

    @abstractmethod
    def next_pipelines(self, data):
        raise NotImplementedError(
            "AutoML has no next_pipelines() implemented!"
        )

    @classmethod
    def tree_impl(cls, data=None):
        raise NotImplementedError(
            "AutoML has no tree() implemented!"
        )

    def fields_to_store_after_use(self):
        raise NotImplementedError(
            "AutoML has no fields_to_store_after_use() implemented!"
        )

    def fields_to_keep_after_use(self):
        raise NotImplementedError(
            "AutoML has no fields_to_keep_after_use() implemented!"
        )
