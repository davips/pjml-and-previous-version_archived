from paje.automl.random import RandomAutoML


class DefaultAutoML(RandomAutoML):
    """
    Always select default hyperparameters.
    """
    def next_dicts(self, forest):
        dicts = [{} for _ in forest]
        return dicts
