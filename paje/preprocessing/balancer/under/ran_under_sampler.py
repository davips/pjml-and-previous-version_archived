from imblearn.under_sampling import RandomUnderSampler
from paje.preprocessing.preprocessing import Preprocessing
from paje.opt.hps import HPTree


class RanUnderSampler(Preprocessing):
    def __init__(self, sampling_strategy):
        self.obj = RandomUnderSampler(sampling_strategy)
        self.newx = self.newy = None

    def fit(self, X, y):
        self.newx, self.newy = self.obj.fit_resample(X, y)

    def transform(self, X=None, y=None):
        return self.newx.copy(), self.newy.copy()

    @staticmethod
    def hps(data):
        return HPTree(data={
            'sampling_strategy': ['c', ['majority', 'not minority',
                                        'not majority', 'all']]
        }, children=None)

    def apply(self, data):
        self.fit(data.data_x, data.data_y)
        data.data_x, data.data_y = self.newx, self.newy

    def use(self, data):
        return data
