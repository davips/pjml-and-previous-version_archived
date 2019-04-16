from imblearn.under_sampling import RandomUnderSampler

from paje.component.component import Component
from paje.base.hps import HPTree


class RanUnderSampler(Component):
    def __init__(self, sampling_strategy):
        self.obj = RandomUnderSampler(sampling_strategy)
        self.newx = self.newy = None

    def apply_impl(self, data):
        X, y = data.xy()
        self.newx, self.newy = self.obj.fit_resample(X, y)
        data.data_x, data.data_y = self.newx, self.newy

    def use_impl(self, data):
        return data

    @staticmethod
    def hps_impl(data=None):
        return HPTree(dic={
            'sampling_strategy': ['c', ['majority', 'not minority',
                                        'not majority', 'all']]
        }, children=[])
