from paje.base.hps import HPTree
from paje.composer.composer import Composer


class Frozen(Composer):
    def __init__(self, component, in_place=False, memoize=False,
                 show_warns=True, **kwargs):
        super().__init__(in_place, memoize, show_warns)

        self.components = [component]
        self.params = kwargs

    def build_impl(self):
        # rnd_state vem de quem chama build()
        self.components = self.components.copy()
        # self.params = self.params.copy()  # TODO: why we needed this here?
        self.components[0].memoize = self.memoize
        self.components[0] = self.components[0].build(**self.dic)

    def freeze_hptree(self):
        aux = {}
        for i in self.params:
            aux[i] = ['c', [self.params[i]]]
        return aux

    def tree_impl(self, data=None):
        return HPTree(dic=self.freeze_hptree(), children=[],
                      name='Frozen: ' + self.components[0].__class__.__name__)

    def __str__(self, depth=''):
        return 'Frozen { ' + str(self.components[0]) + ' }'
