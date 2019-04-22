from typing import List, Dict

from paje.base.component import Component


class Pipeline(Component):
    #TODO: An empty Pipeline may return perfect predictions.
    def init_impl(self, components: List[Component], hyperpar_dicts: [Dict] = None):
        self.components = components
        if hyperpar_dicts is None: hyperpar_dicts = [{} for _ in components]
        zipped = zip(self.components, hyperpar_dicts)
        self.instances = [component(**hyperpar_dict) for component, hyperpar_dict in zipped]

    def apply_impl(self, data):
        for instance in self.instances:
            data = instance.apply(data)  # useless assignment if aux is set to be inplace
        return data

    def use_impl(self, data):
        for instance in self.instances:
            data = instance.use(data)  # useless assignment if aux is set to be inplace
        return data

    @classmethod
    def hyperpar_spaces_tree_impl(cls, data=None):
        raise NotImplementedError("Pipeline has no method hyper_spaces_tree() implemented,",
                                  "because it would depend on the constructor parameters.",
                                  "hyper_spaces_forest() should be called instead!")

    def hyperpar_spaces_forest(self, data=None):
        forests = [instance.hyperpar_spaces_forest(data) for instance in self.instances]
        return sum(forests, [])  # flatten out the lists
