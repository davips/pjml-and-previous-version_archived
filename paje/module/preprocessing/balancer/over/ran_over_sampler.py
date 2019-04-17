from imblearn.over_sampling import RandomOverSampler
from paje.base.hps import HPTree
from paje.component import Component


class RanOverSampler(Component):
    def init_impl(self, sampling_strategy):
        self.obj = RandomOverSampler(sampling_strategy)
        self.newx = self.newy = None

    def apply_impl(self, data):
        X, y = data.xy()
        self.newx, self.newy = self.obj.fit_resample(X, y)
        data.data_x, data.data_y = self.newx, self.newy

    def use_impl(self, data):
        return data

    @classmethod
    def hps_impl(cls, data=None):
        return HPTree(dic={
            'sampling_strategy': ['c', ['not minority',
                                        'not majority', 'all']]
        }, children=[])
