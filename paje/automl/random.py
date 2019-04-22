from paje.automl.automl import AutoML
from paje.base.hps import HPTree


class RandomAutoML(AutoML):
    def next_hyperpar_dicts(self):
        dicts = [self.tree_to_dict(tree) for tree in self.forest]
        return dicts

    def tree_to_dict(self, tree: HPTree):
        return {} # TODO
