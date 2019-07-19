from paje.base.data import Data
from paje.base.hp import CatHP
from paje.base.hps import ConfigSpace
from paje.ml.element.element import Element
from paje.util.distributions import choice


class Reduce(Element):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.field = self.config['field']

    def apply_impl(self, data):
        return self.use_impl(data)

    def use_impl(self, data):
        g = data.g + [data._get(self.field)]
        return data.updated(self, g=g)

    @classmethod
    def tree_impl(cls):
        hps = [
            CatHP('field', choice, itens=['x','y','z'])   # TODO: cirar funcao no data
        ]
        return ConfigSpace(name=cls.__name__, hps=hps)
