import copy
import random

import numpy as np

from paje.automl.automl import AutoML
from paje.composer.pipeline import Pipeline
from paje.util.distributions import SamplingException
from paje.evaluator.evaluator import Evaluator
from paje.evaluator.metrics import Metrics

class RandomAutoML(AutoML):

    def __init__(self,
                 preprocessors=None,
                 modelers=None,
                 storage_for_components=None,
                 static=True,
                 fixed=True,
                 max_depth=5,
                 repetitions=0,
                 random_state=0,
                 method="all",
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
        evaluator = Evaluator(Metrics.accuracy, "cv", 10,
                              random_state=random_state)
        AutoML.__init__(self, evaluator, random_state, **kwargs)

        self.best_eval = -999999
        if static and not fixed:
            self.error('static and not fixed!')
        if static and repetitions > 0:
            self.error('static and repetitions > 0!')
        self.max_depth = max_depth
        self.static = static
        self.fixed = fixed
        self.repetitions = repetitions

        self.modelers = modelers
        self.preprocessors = preprocessors
        self.storage_for_components = storage_for_components

        self.current_eval = None

        if static:
            if len(self.modelers) > 1:
                self.warning('Multiple modelers given in static mode.')
            self.static_pipeline = self.preprocessors + self.modelers
            if max_depth < len(self.static_pipeline):
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
