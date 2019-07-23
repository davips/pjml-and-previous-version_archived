import numpy

from paje.base.hp import CatHP
from paje.base.hps import ConfigSpace
from paje.ml.element.element import Element
from paje.util.distributions import choice


class Summ(Element):
    _functions = {
        'mean': numpy.mean
    }

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._function = self._functions[self.config['function']]
        self._field = self.config['field']

    def apply_impl(self, data):
        return self.use_impl(data)

    def use_impl(self, data):
        stack, aux = data.S.pop()
        return data.updated(
            self,
            S=stack,
            **{self._field: numpy.array([self._function(aux)])})

    @classmethod
    def tree_impl(cls):
        hps = [
            CatHP('function', choice, itens=['mean'])
        ]
        return ConfigSpace(name=cls.__name__, hps=hps)

    @classmethod
    def isdeterministic(cls):
        return True
