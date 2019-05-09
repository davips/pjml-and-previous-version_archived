from paje.base.component import Component


class Pipeline(Component):
    # TODO: An empty Pipeline may return perfect predictions.
    def __init__(self, components=None, in_place=False, memoize=False,
                 show_warns=True):
        super().__init__(in_place, memoize, show_warns)
        if components is None:
            components = []
        self.components = components
        self.random_state = 0

    def instantiate_impl(self):
        """
        The only parameter is dics with the dic of each component.
        :param dics
        :return:
        """
        dics = [{} for _ in self.components]  # Default value
        if 'dics' in self.dic:
            dics = self.dic['dics']
        if 'random_state' in self.dic:
            self.random_state = self.dic['random_state']

        zipped = zip(range(0, len(self.components)), dics)
        for idx, dic in zipped:
            dic['random_state'] = self.random_state
            self.components[idx] = self.components[idx].instantiate(**dic)
            # component.instantiate(**dic)

    def apply_impl(self, data):
        for component in self.components:
            # useless assignment if it is set to be inplace
            data = component.apply(data)
        return data

    def use_impl(self, data):
        for component in self.components:
            data = component.use(data)
        return data

    @classmethod
    def tree_impl(cls, data=None):
        raise NotImplementedError("Pipeline has no method tree() \
                                      implemented, because it would depend on the \
                                      constructor parameters. \
                                      forest() \
                                      should be called instead!")

    def forest(self, data=None):  # previously known as hyperpar_spaces_forest
        bigger_forest = []
        for component in self.components:
            if isinstance(component, Pipeline):
                forest = list(map(
                    lambda x: x.forest(data),
                    component.components
                ))
            else:
                forest = component.tree(data)
            bigger_forest.append(forest)
        return bigger_forest

    def __str__(self, depth=''):
        depth += '    '
        strs = [component.__str__(depth) for component in self.components]
        return "Pipeline {\n" + depth + ("\n" + depth).join(
            str(x) for x in strs) + "\n}\n"

    def handle_storage(self, data):
        # TODO: replicate this method to other nesting modules (Chooser...),
        #  not only
        #  Pipeline and AutoML
        return self.apply_impl(data)
