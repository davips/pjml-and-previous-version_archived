import random

from paje.automl.automl import AutoML


class RandomAutoML(AutoML):

    def next_hyperpar_dicts(self, forest):
        dics = []
        if isinstance(forest, list):
            for item in forest:
                dics.append(self.next_hyperpar_dicts(item))
            return dics
        else:
            return forest.tree_to_dict()

    def choose_modules(self):
        # TODO:
        #  static ok
        #  fixed ok
        #  no repetitions ok
        #  repetitions ok
        take = self.max_depth if self.fixed else random.randint(1, self.max_depth)
        preprocessors = self.preprocessors * (self.repetitions + 1)
        random.shuffle(preprocessors)
        return preprocessors[:take] + [random.choice(self.modelers)]


