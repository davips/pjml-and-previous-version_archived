from paje.base.chain import Chain
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
        val = data.get(self.field)
        return data.updated(self, S=Chain(val, data.S, idx=0))

    @classmethod
    def tree_impl(cls):
        # TODO: cirar funcao no data
        hps = [
            CatHP('field', choice, itens=['X','Y','Z'])
        ]
        return ConfigSpace(name=cls.__name__, hps=hps)

    @classmethod
    def isdeterministic(cls):
        return True
