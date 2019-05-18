from paje.base.component import Component


class Resampler(Component):
    def apply_impl(self, data):
        X, y = self.model.fit_resample(*data.xy)
        return data.update(X=X, y=y)

    def use_impl(self, data):
        return data
