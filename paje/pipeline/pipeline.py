from paje.base.component import Component


class Pipeline(Component):
    # TODO: An empty Pipeline may return perfect predictions.
    def __init__(self, components=None, hyperpar_dicts=None, just_for_tree=False,
                 in_place=False, memoize=False, show_warnings=True, **kwargs):
        super().__init__(in_place, memoize, show_warnings, kwargs)
        if components is None:
            components = []
        self.components = components
        self.memoize = memoize
        self.random_state = kwargs['random_state'] if 'random_state' in kwargs else 0

        if hyperpar_dicts is None:
            self.hyperpar_dicts = [{} for _ in components]
        else:
            self.hyperpar_dicts = hyperpar_dicts
        self.just_for_tree = just_for_tree
        self.instances = []
        self.instantiate_components()

    def apply_impl(self, data):
        for instance in self.instances:
            # useless assignment if it is set to be inplace
            data = instance.apply(data)
        return data

    def use_impl(self, data):
        for instance in self.instances:
            data = instance.use(data)
        return data

    @classmethod
    def tree_impl(cls, data=None):
        raise NotImplementedError("Pipeline has no method hyper_spaces_tree() \
                                  implemented, because it would depend on the \
                                  constructor parameters. \
                                  hyper_spaces_forest() \
                                  should be called instead!")

    def forest(self, data=None):  # previously known as hyperpar_spaces_forest
        bigger_forest = []
        for instance in self.instances:
            if isinstance(instance, Pipeline):
                forest = list(map(
                    lambda x: x.forest(data),
                    instance.instances
                ))
            else:
                forest = instance.tree(data)
            bigger_forest.append(forest)
        return bigger_forest

    def __str__(self, depth=''):
        depth += '    '
        strs = [instance.__str__(depth) for instance in self.instances]
        return super().__str__() + "\n" + depth + ("\n" + depth).join(
            str(x) for x in strs)

    def instantiate_components(self):
        if len(self.instances) > 0:
            print(self.instances)
            raise Exception('instances list should be empty!')
        zipped = zip(self.components, self.hyperpar_dicts)
        instance = None
        for component, hyperpar_dict in zipped:
            if isinstance(component, Pipeline):
                if not self.just_for_tree:
                    component = Pipeline(component.components, hyperpar_dict,
                                         memoize=self.memoize,
                                         random_state=self.random_state)
                instance = component
            else:
                try:
                    if not self.just_for_tree:
                        instance = component(**hyperpar_dict, memoize=self.memoize,
                                             random_state=self.random_state)
                    else:
                        instance = component  # gambiarra para evitar instanciar sem argumentos (DRFTAG quebra, por exemplo)
                except:
                    self.error(component)

            self.instances.append(instance)

    def handle_storage(self, data):
        # TODO: replicate this method to other nesting modules, not only
        # Pipeline and AutoML
        return self.apply_impl(data)
