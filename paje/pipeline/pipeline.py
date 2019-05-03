from typing import List, Dict

from paje.base.component import Component


class Pipeline(Component):
    # TODO: An empty Pipeline may return perfect predictions.
    def init_impl(self, components: List, hyperpar_dicts: [Dict] = None):
        self.components = components
        self.hyperpar_dicts = [{} for _ in components] if hyperpar_dicts is None \
            else hyperpar_dicts
        self.instantiate_components()

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
        bigger_forest = []
        self.instantiate_components(just_for_tree=True)
        for instance in self.instances:
            if isinstance(instance, Pipeline):
                forest = list(map(
                    lambda x: x.hyperpar_spaces_forest(data),
                    instance.instances
                ))
            else:
                forest = instance.hyperpar_spaces_forest(data)
            bigger_forest.append(forest)
        return bigger_forest

    def __str__(self, depth=''):
        depth += '    '
        strs = [instance.__str__(depth) for instance in self.instances]
        return super().__str__() + "\n" + depth + ("\n" + depth).join(str(x) for x in strs)

    def instantiate_components(self, just_for_tree=False):
        self.instances = []
        zipped = zip(self.components, self.hyperpar_dicts)
        for component, hyperpar_dict in zipped:
            if isinstance(component, Pipeline):
                if not just_for_tree:
                    component = Pipeline(component.components, hyperpar_dict)
                instance = component
            else:
                try:
                    instance = component(**hyperpar_dict)
                except:
                    self.error(component)

            self.instances.append(instance)
