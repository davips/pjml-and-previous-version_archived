from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler

from paje.base.hp import CatHP
from paje.base.hps import ConfigSpace
from paje.ml.element.preprocessing.unsupervised.feature.scaler.scaler import \
    Scaler
from paje.util.distributions import choice


class Equalization(Scaler):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MinMaxScaler(**self.config)

    @classmethod
    def tree_impl(cls):
        hps = {
            'feature_range': CatHP(choice, items=[(-1, 1), (0, 1)])
        }
        return ConfigSpace(name=cls.__name__, hps=hps)
