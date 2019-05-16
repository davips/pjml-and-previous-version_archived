from paje.automl.random import RandomAutoML


class DefaultAutoML(RandomAutoML):
    """
    Always select default hyperparameters.
    """
    def next_args(self, forest):
        dicts = [{} for _ in forest]
        return dicts
