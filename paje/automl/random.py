import copy
import random

import numpy as np

from paje.automl.automl import AutoML
from paje.composer.pipeline import Pipeline
from paje.util.distributions import SamplingException
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics

class RandomAutoML(AutoML):

    DEFAULT_MODELERS = []
    DEFAULT_PREPROCESSORS = []

    def __init__(self,
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

        AutoML.__init__(self, **kwargs)

        # Attributes set in the build_impl.
        # These attributes identify uniquely AutoML.
        # This structure is necessary because the AutoML is a Component and it
        # could be used into other Components, like the Pipeline one.
        self.max_depth = 2
        self.static = False
        self.fixed = False
        self.repetitions = 0
        self.modelers = self.DEFAULT_MODELERS
        self.preprocessors = self.DEFAULT_PREPROCESSORS

        # Other class attributes.
        # These attributes can be set here or in the build_impl method. They
        # should not influence the AutoML final result.
        self.storage_for_components = storage_for_components

        # Class internal attributes
        # Attributes that were not parameterizable
        self.best_eval = -999999
        self.current_eval = None
        self.static_pipeline = None

    def build_impl(self):
        """ TODO the docstring documentation
        """
        # The 'self' is a copy, because of the Component.build().
        # Be careful, the copy made in the parent (Component) is
        # shallow (copy.copy(self)).
        # See more details in the Component.build() method.

        self.evaluator = Evaluator(Metrics.accuracy, "cv", 10,
                                   random_state=self.random_state)

        # making the build_impl of father
        super().build_impl(self)

        # maybe it is not necessary
        self.__dict__.update(self.dic)

        # TODO: check if it is necessary
        if self.static and not self.fixed:
            self.error('static and not fixed!')
        if self.static and self.repetitions > 0:
            self.error('static and repetitions > 0!')

        # TODO: check if it is necessary
        if self.static:
            if len(self.modelers) > 1:
                self.warning('Multiple modelers given in static mode.')
            self.static_pipeline = self.preprocessors + self.modelers
            if self.max_depth < len(self.static_pipeline):
                self.warning('max_depth lesser than given fixed pipeline!')
        random.seed(self.random_state)

    def next_pipelines(self, data):
        modules = self.static_pipeline if self.static else self.choose_modules()
        self.curr_pipe = Pipeline(modules, show_warns=self.show_warns,
                                  storage=self.storage_for_components)
        tree = self.curr_pipe.tree(data)
        try:
            args = self.next_args(tree)
        except SamplingException as e:
            print(' ========== Pipe:\n', self.curr_pipe)
            raise Exception(e)
        args.update(random_state=self.random_state)
        # print('tree...\n', tree)
        # print(' args...\n', args)
        self.curr_pipe = self.curr_pipe.build(**args)
        return [self.curr_pipe]

    def next_args(self, forest):
        return forest.tree_to_dict()

    def choose_modules(self):
        # DONE:
        #  static ok
        #  fixed ok
        #  no repetitions ok
        #  repetitions ok
        take = self.max_depth \
            if self.fixed else random.randint(0, self.max_depth)
        preprocessors = self.preprocessors * (self.repetitions + 1)
        random.shuffle(preprocessors)
        return preprocessors[:take] + [random.choice(self.modelers)]
        # tmp = []
        # for preprocessor in self.preprocessors:
        #     for _ in range(0, self.repetitions):
        #         tmp.append(copy.deepcopy(preprocessor))
        # random.shuffle(tmp)
        # return tmp[:take] + [random.choice(self.modelers)]

    def process_step(self, eval_result):
        self.current_eval = np.mean(eval_result)
        if self.current_eval is not None\
           and self.current_eval > self.best_eval:
            self.best_eval = self.current_eval
            self.best_pipe = self.curr_pipe

    def get_best_pipeline(self):
        return self.best_pipe

    def get_current_eval(self):
        return self.current_eval

    def get_best_eval(self):
        return self.best_eval
