from paje.base.component import Component


class Pipeline(Component):

    # components is like this --> [(obj, {}), (obj, {}), (obj, {})]
    def init_impl(self, components):
        self.components = components
        self.obj_comp = []

    def apply_impl(self, data):
        self.obj_comp = []
        # print(self.components)
        for obj, param in self.components:
            # print(param)
            aux = obj(**param)  # TODO: here it is possible to choose if obj handles Data as inplace or not
            data = aux.apply(data)  # useless assignment if aux is set to be inplace
            self.obj_comp.append(aux)
        return data

    def use_impl(self, data):
        for obj in self.obj_comp:
            # print(data)
            data = obj.use(data)  # useless assignment if aux is set to be inplace
        return data

    @classmethod
    def hps_impl(cls, data=None):
        raise NotImplementedError("Method hps should be implemented!")
