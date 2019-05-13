from paje.base.component import Component
from paje.base.hps import HPTree

class Freeze(Component):
    def __init__(self, component, in_place=False, memoize=False,
                 show_warns=True, **kwargs):
        super().__init__(in_place, memoize, show_warns)

        self.component = component
        self.params = kwargs

    def instantiate_impl(self):
        self.component = self.component.instantiate(**self.params)

    def apply_impl(self, data):
        return self.component.apply(data)

    def use_impl(self, data):
        return self.component.use(data)

    @classmethod
    def tree_impl(cls, data=None):
        raise NotImplementedError("Not implemented")

    def freeze_hptree(self):
        aux = {}
        for i in self.params:
            aux[i] = ['c', [self.params[i]]]
        print(aux)
        return aux

    def forest(self, data=None):
        return HPTree(dic=self.freeze_hptree(), children=[])
