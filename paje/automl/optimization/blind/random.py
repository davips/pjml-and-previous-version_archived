import random

import numpy as np

from paje.automl.automl import AutoML
from paje.automl.composer.pipeline import Pipeline
from paje.util.distributions import SamplingException
from paje.evaluator.evaluator import EvaluatorClassif


class RandomAutoML(AutoML):
    def __init__(self,
                 preprocessors,
                 modelers,
                 storage_for_components=None,
                 **kwargs):
        """
        AutoML
        :param preprocessors: list of modules for balancing,
            noise removal, sampling etc.
        :param modelers: list of modules for prediction
            (classification or regression etc.)
        :param repetitions: how many times can a module appear
            in a pipeline
        :param method: TODO
        :param max_iter: maximum number of pipelines to evaluate
        :param max_depth: maximum length of a pipeline
        :param static: are the pipelines generated always exactly
            as given by the ordered list preprocessors + modelers?
        :param fixed: are the pipelines generated always with
            length max(max_depth, len(preprocessors + modelers))?
        :param random_state: TODO
        :return:
        """

        AutoML.__init__(self,
                        components=preprocessors + modelers,
                        evaluator=EvaluatorClassif(),
                        **kwargs)

        # These attributes identify uniquely AutoML.
        # This structure is necessary because the AutoML is a Component and it
        # could be used into other Components, like the Pipeline one.
        # build_impl()
        self.repetitions = 0  # by default, no repetitions was be performed
        self.pipe_length = 2  # by default, the pipeline has two components
        # __init__()
        self.modelers = modelers
        self.preprocessors = preprocessors

        if not isinstance(modelers, list) or \
                not isinstance(preprocessors, list):
            print(modelers)
            print(preprocessors)
            raise TypeError("The modelers/preprocessors must be list.")

        if not modelers:
            raise ValueError("The list length must be greater than one.")

        # Other class attributes.
        # These attributes can be set here or in the build_impl method. They
        # should not influence the AutoML final result.
        self.storage_for_components = storage_for_components

        # Class internal attributes
        # Attributes that were not parameterizable
        self.best_eval = float('-Inf')
        self.best_pipe = None
        self.curr_eval = None
        self.curr_pipe = None

    def build_impl(self, **args_set):
        """ TODO the docstring documentation
        """
        # The 'self' is a copy.
        # Be careful, the copy made in the parent (Component) is
        # shallow (copy.copy(self)).
        # See more details in the Component.build() method.

        self.__dict__.update(self.args_set)
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        if self.pipe_length < 1:
            raise ValueError("The 'pipe_length' must be greater than 0.")

        #TODO: was this IF correct?  ass. Davi
        if self.repetitions < 0:
            print('self.repetitions', self.repetitions)
            raise ValueError("The 'repetitions' must be a non-negative int.")

    def next_pipelines(self, data):
        """ TODO the docstring documentation
        """
        components = self.choose_modules()
        self.curr_pipe = Pipeline(components, show_warns=self.show_warns,
                                  storage=self.storage_for_components)
        tree = self.curr_pipe.tree()

        try:
            args = tree.tree_to_dict()
        except SamplingException as exc:
            print(' ========== Pipe:\n', self.curr_pipe)
            raise Exception(exc)

        args['random_state'] = self.random_state
        self.curr_pipe = self.curr_pipe.build(**args)
        return [self.curr_pipe]

    def choose_modules(self):
        """ TODO the docstring documentation
        """
        take = np.random.randint(0, self.pipe_length)

        preprocessors = self.preprocessors * (self.repetitions + 1)
        np.random.shuffle(preprocessors)
        return preprocessors[:take] + [np.random.choice(self.modelers)]

    def process_step(self, eval_result):
        """ TODO the docstring documentation
        """
        self.curr_eval = eval_result[0][1] or 0
        if self.curr_eval is not None \
                and self.curr_eval > self.best_eval:
            self.best_eval = self.curr_eval
            self.best_pipe = self.curr_pipe

    def get_best_pipeline(self):
        """ TODO the docstring documentation
        """
        return self.best_pipe

    def get_current_eval(self):
        """ TODO the docstring documentation
        """
        return self.curr_eval

    def get_best_eval(self):
        """ TODO the docstring documentation
        """
        return self.best_eval
