""" TODO the docstring documentation
"""
# from python
from abc import ABC, abstractmethod

# from other packages
# import numpy as np

# from paje
from paje.base.component import Component


class AutoML(Component, ABC):
    """ TODO the docstring documentation
    """
    def __init__(self,
                 evaluator,
                 n_jobs=1,
                 verbose=True,
                 max_iter=10,
                 random_state=0,
                 storage=None,
                 show_warns=True,
                 **kwargs):
        super().__init__(storage, show_warns, **kwargs)
        """ TODO the docstring documentation
        """
        self.evaluator = evaluator
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

        # internal class atributes
        self.model = None
        self.all_eval_results = []
        self.fails, self.locks, self.successes, self.total = 0, 0, 0, 0
        self.current_iteration = 0

    def build_impl(self):
        """ TODO the docstring documentation
        """
        # It is not clear if this class is instantiable yet.
        return self

    def eval_pipelines_par(self, pipelines, data, eval_results):
        """ TODO the docstring documentation
        """

    def eval_pipelines_seq(self, pipelines, data, eval_results):
        """ TODO the docstring documentation
        """
        for pipe in pipelines:
            self.total += 1
            if self.verbose:
                print(pipe)
            eval_result = self.evaluator.eval(pipe, data)
            if pipe.failed:
                self.fails += 1
            elif pipe.locked:
                self.locks += 1
            else:
                self.successes += 1
            eval_results.append(eval_result)

    def apply_impl(self, data):
        """ TODO the docstring documentation
        """
        for iteration in range(1, self.max_iter+1):
            self.current_iteration = iteration
            if self.verbose:
                print("####------##-----##-----##-----##-----##-----####")
                print("Current iteration = ", self.current_iteration)
            # Evaluates current hyperparameter (space-values) combination.
            pipelines = self.next_pipelines(data)

            # this variable saves the results of the current iteration
            eval_result = []
            if self.n_jobs > 1:
                # Runs all pipelines generated in this iteration (parallelly)
                # and put the results in the eval_result variable
                self.eval_pipelines_par(pipelines, data, eval_result)
            else:
                # Runs all pipelines generated in this iteration (sequentially)
                # and put the results in the eval_result variable
                self.eval_pipelines_seq(pipelines, data, eval_result)

            # This attribute save all results
            self.all_eval_results.append(eval_result)

            self.process_step(eval_result)
            if self.verbose:
                print("Current ...............: ", self.get_current_eval())
                print("Best ..................: ", self.get_best_eval())
                print("Locks/fails/successes .: {0}/{1}/{2}".format(
                    self.locks, self.fails, self.successes))
                print("####------##-----##-----##-----##-----##-----####\n")

        self.process_all_steps(self.all_eval_results)

        self.model = self.get_best_pipeline()
        if self.verbose:
            print("Best pipeline found:")
            print(self.model)

        return self.model.apply(data)

    def use_impl(self, data):
        """ TODO the docstring documentation
        """
        return self.model.use(data)

    def get_best_eval(self):
        """ TODO the docstring documentation
        """

    def get_current_eval(self):
        """ TODO the docstring documentation
        """

    def process_step(self, eval_result):
        """ TODO the docstring documentation
        """

    def process_all_steps(self, eval_results):
        """ TODO the docstring documentation
        """

    @abstractmethod
    def get_best_pipeline(self):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no get_best_pipeline() implemented!"
        )

    @abstractmethod
    def next_pipelines(self, data):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no next_pipelines() implemented!"
        )

    @classmethod
    def tree_impl(cls, data=None):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no tree() implemented!"
        )

    def fields_to_store_after_use(self):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no fields_to_store_after_use() implemented!"
        )

    def fields_to_keep_after_use(self):
        """ TODO the docstring documentation
        """
        raise NotImplementedError(
            "AutoML has no fields_to_keep_after_use() implemented!"
        )
