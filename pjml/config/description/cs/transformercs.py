from pjml.config.description.cs.abc.componentcs import ComponentCS


class TransformerCS(ComponentCS):
    """Complete settings for a component (a component is a set of
    transformers, e.g. the component KNN represents the set of all k-NN
    transformers: KNN(k=1), KNN(k=3), ...

    Parameters
    ----------
    nodes
        List of internal nodes. Only one is sampled at a time.
    name
        Name (usually the Python class) of the component.
    path
        Path (usually the Python module) of the component.
    """

    def __init__(self, name=None, path=None, nodes=None):
        print(2222222222222222222)
        if nodes is None:
            raise Exception('TransformerCS should have a list of nodes!')
        super().__init__(name, path, None, nodes)

    def _sample_cfg(self):
        return {}

    def identified(self, name, path):
        return self.__class__(name, path, self.nodes)

    def updated(self, nodes):
        return self.__class__(self.name, self.path, nodes)
