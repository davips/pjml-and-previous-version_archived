from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler

from paje.base.hp import CatHP
from paje.base.hps import ConfigSpace
from paje.ml.element.preprocessing.unsupervised.feature.scaler.scaler import \
    Scaler


class Equalization(Scaler):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MinMaxScaler(**self.config)

    @classmethod
    def tree_impl(cls, data=None):
        hps = [
            CatHP('feature_range', cls.sampling_function, items=[(-1, 1), (0, 1)])
        ]
        return ConfigSpace(name=cls.__name__, hps=hps)

    @staticmethod
    def sampling_function(items):
        idx = randint(0, len(items))
        return items[idx]
