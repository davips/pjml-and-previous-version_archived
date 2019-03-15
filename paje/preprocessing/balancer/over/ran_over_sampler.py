from imblearn.over_sampling import RandomOverSampler
from paje.base.hps import HPTree
from paje.base.component import Component


class RanOverSampler(Component):
    def __init__(self, sampling_strategy):
        self.obj = RandomOverSampler(sampling_strategy)
        self.newx = self.newy = None

    def apply(self, data):
        X, y = data.xy()
        self.newx, self.newy = self.obj.fit_resample(X, y)
        data.data_x, data.data_y = self.newx, self.newy

    def use(self, data):
        return data

    @staticmethod
    def hps(data=None):
        return HPTree(data={
            'sampling_strategy': ['c', ['not minority',
                                        'not majority', 'all']]
        }, children=[])
