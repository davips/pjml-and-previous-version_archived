from paje.base.component import Component
from paje.base.data import Data


class Resampler(Component):
    def touched_fields(self):
        return None

    def apply_impl(self, data):
        # TODO: generalize this to resample all fields (xyzuvwpq...)
        X, y = self.model.fit_resample(*data.Xy)
        return data.updated(self, X=X, y=y)

    def use_impl(self, data):
        return data
