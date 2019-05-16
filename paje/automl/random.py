import copy
import random

from paje.automl.automl import AutoML
from paje.composer.pipeline import Pipeline


class RandomAutoML(AutoML):

    def __init__(self, preprocessors=None, modelers=None,
                 max_iter=5, static=True, fixed=True, max_depth=5,
                 repetitions=0, method="all", verbose=True, random_state=0,
                 in_place=False, memoize=False, show_warns=True):
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
        AutoML.__init__(self, preprocessors=preprocessors, modelers=modelers,
                        verbose=verbose,
                        random_state=random_state, in_place=in_place,
                        memoize=memoize, show_warns=show_warns)

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
        modules = self.static_pipeline if self.static else self.choose_modules()
        self.curr_pipe = Pipeline(modules, in_place=self.in_place,
                                  memoize=self.memoize,
                                  show_warns=self.show_warns)
        forest = self.curr_pipe.tree(data)
        # print('fores\n', forest)
        # print(' to dic\n', self.next_args(forest))
        self.curr_pipe = self.curr_pipe.build(**self.next_args(forest),
                                              random_state=self.random_state)
        return [self.curr_pipe]

    def next_args(self, forest):  # previously known as next_hyperpar_dicts
        # Defines search space (space of hyperparameter spaces).
        # dics = []
        # if isinstance(forest, list):
        #     for item in forest:
        #         dics.append(self.next_dicts(item))
        #     return dics
        # else:
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

    def process(self, errors):
        if errors[0] is not None and errors[0] < self.best_error:
            self.best_error = errors[0]
            self.best_pipe = self.curr_pipe

    def best(self):
        return self.best_pipe
