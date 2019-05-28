from paje.base.component import Component


class Composer(Component):
    # TODO: An empty Pipeline/composer may return perfect predictions.
    def __init__(self, components=None, storage=None, show_warns=True):
        super().__init__(storage, show_warns)
        if components is None:
            components = []
        self.components = components
        self.random_state = 0
        self.mytree = None
        self.model = 42

    def apply_impl(self, data):
        for component in self.components:
            data = component.apply(data)
        return data

    def use_impl(self, data):
        for component in self.components:
            data = component.use(data)
        return data
