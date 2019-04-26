from paje.automl.automl import AutoML


class DefaultAutoML(AutoML):
    """
    Always select default hyperparameters.
    """
    def next_hyperpar_dicts(self, forest):
        dicts = [{} for _ in forest]
        return dicts
