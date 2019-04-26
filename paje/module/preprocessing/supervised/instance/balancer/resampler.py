from paje.base.component import Component


class Resampler(Component):
    def apply_impl(self, data):
        data.data_x, data.data_y = self.model.fit_resample(*data.xy())
        return data

    def use_impl(self, data):
        return data
