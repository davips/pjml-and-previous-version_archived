from composer.frozen import Frozen


class Melting(Frozen):
    def build_impl(self):
        # rnd_state vem de quem chama build()
        self.components = self.components.copy()
        # self.params = self.params.copy()  # TODO: why we needed this here?
        self.dic.update(self.params)
        self.components[0] = self.components[0].build(**self.dic)

    def tree_impl(self, data=None):
        return self.components[0].tree()
