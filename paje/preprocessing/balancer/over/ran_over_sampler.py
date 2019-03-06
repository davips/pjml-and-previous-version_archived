from imblearn.over_sampling import RandomOverSampler
from paje.preprocessing.preprocessing import Preprocessing
from paje.opt.hps import HPTree


class RanOverSampler(Preprocessing):
    def __init__(self, sampling_strategy):
        self.obj = RandomOverSampler(sampling_strategy)
        self.newx = self.newy = None

    def fit(self, X, y):
        self.newx, self.newy = self.obj.fit_resample(X, y)

    def transform(self, X=None, y=None):
        return self.newx.copy(), self.newy.copy()

    def hps(self, data):
        return HPTree(data={
            'sampling_strategy': ['c', ['majority', 'not minority',
                                        'not majority', 'all']]
        }, children=None)

    def apply(self, data):
        self.fit(data.data_x, data.data_y)
        data.data_x, data.data_y = self.transfrom()

    def use(self, data):
        return data
