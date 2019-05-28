from paje.base.hps import HPTree
from paje.composer.composer import Composer


class Frozen(Composer):
    def fields_to_store_after_use(self):
        return self.components[0].fields_to_store_after_use()

    def fields_to_keep_after_use(self):
        return self.components[0].fields_to_keep_after_use()

    def __init__(self, component, in_place=False, storage=None,
                 show_warns=True, **kwargs):
        super().__init__(in_place, storage, show_warns)

        self.components = [component]
        self.params = kwargs

    def build_impl(self):
        # rnd_state vem de quem chama build()
        self.components = self.components.copy()
        # self.params = self.params.copy()  # TODO: why we needed this here?
        self.components[0] = self.components[0].build(**self.dic)

    def freeze_hptree(self):
        aux = {}
        for i in self.params:
            aux[i] = ['c', [self.params[i]]]
        return aux

    def tree_impl(self, data=None):
        return HPTree(dic=self.freeze_hptree(), children=[],
                      name=self.name + ': ' + self.components[0].name)

    def __str__(self, depth=''):
        return self.name + ' { ' + str(self.components[0]) + ' }'
