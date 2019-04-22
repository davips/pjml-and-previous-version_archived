from paje.automl.automl import AutoML


class DefaultAutoML(AutoML):
    def next_hyperpar_dicts(self):
        dicts = [{} for _ in self.forest]
        return dicts
