import random

from paje.automl.automl import AutoML
from paje.pipeline.pipeline import Pipeline

class RandomAutoML(AutoML):

    def __init__(self, preprocessors=None, modelers=None,
                 max_iter=2, static=True, fixed=True, max_depth=5,
                 repetitions=0, method="all", random_state=0,
                 in_place=False, memoize=False, show_warnings=True):
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
        AutoML.__init__(self, preprocessors, modelers, random_state,
                        in_place, memoize, show_warnings)

        self.best_error = 9999999
        if static and not fixed:
            self.error('static and not fixed!')
        if static and repetitions > 0:
            self.error('static and repetitions > 0!')
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.static = static
        self.fixed = fixed
        self.repetitions = repetitions
        if static:
            if len(self.modelers) > 1:
                self.warning('Multiple modelers given in static mode.')
            self.static_pipeline = self.preprocessors + self.modelers
            if max_depth < len(self.static_pipeline):
                self.warning('max_depth lesser than given fixed pipeline!')
        random.seed(self.random_state)

    def next_pipelines(self, data):
        modules = self.static_pipeline\
                if self.static else self.choose_modules()
        forest = Pipeline(modules).hyperpar_spaces_forest(data)

        self.curr_pipe = Pipeline(modules, self.next_hyperpar_dicts(forest),
                                  self.random_state)
        return [self.curr_pipe]

    def next_hyperpar_dicts(self, forest):
        # Defines search space (space of hyperparameter spaces).
        dics = []
        if isinstance(forest, list):
            for item in forest:
                dics.append(self.next_hyperpar_dicts(item))
            return dics
        else:
            return forest.tree_to_dict()

    def choose_modules(self):
        # DONE:
        #  static ok
        #  fixed ok
        #  no repetitions ok
        #  repetitions ok
        take = self.max_depth\
                if self.fixed else random.randint(0, self.max_depth)
        preprocessors = self.preprocessors * (self.repetitions + 1)
        random.shuffle(preprocessors)
        return preprocessors[:take] + [random.choice(self.modelers)]

    def process(self, errors):
        if errors[0] < self.best_error:
            self.best_error = errors[0]
            self.best_pipe = self.curr_pipe

    def best(self):
        return self.best_pipe
