from paje.automl.random import RandomAutoML


class DefaultAutoML(RandomAutoML):
    """
    Always select default hyperparameters.
    """
    def next_args(self, tree):
        dicts = [{} for _ in tree]
        return dicts
