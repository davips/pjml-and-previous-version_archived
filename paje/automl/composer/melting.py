from paje.automl.composer.frozen import Frozen


class Melting(Frozen):
    def build_impl(self):
        # rnd_state vem de quem chama build()
        self.components = self.components.copy()
        # self.params = self.params.copy()  # TODO: why we needed this here?
        self.dic.update(self.params)
        self.components[0] = self.components[0].build(**self.dic)

    def tree_impl(self):
        # TODO: the tree reported here is bigger than if we considered
        #  'params' here to restrict the tree. If one just wants to override
        #  the random_state, that's ok, but if one wants to freeze a greater
        #  portion of the tree, it should modify the tree here to avoid
        #  useless search of the space by automl.
        tree = self.components[0].tree()
        tree.name = self.name + tree.name
        return tree
