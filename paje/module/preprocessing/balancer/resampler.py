from imblearn.under_sampling import RandomUnderSampler

from paje.component import Component
from paje.base.hps import HPTree


class Resampler(Component):
    def apply_impl(self, data):
        data.data_x, data.data_y = self.model.fit_resample(*data.xy())
        return data

    def use_impl(self, data):
        return data
