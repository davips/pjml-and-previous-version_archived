from paje.component.component import Component


class Resampler(Component):
    def apply_impl(self, data):
        # TODO: generalize this to resample all fields (xyzuvwpq...)
        X, y = self.model.fit_resample(*data.Xy)
        return data.updated(self, X=X, y=y)

    def use_impl(self, data):
        return data
