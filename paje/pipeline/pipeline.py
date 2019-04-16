from paje.component.component import Component


class Pipeline(Component):

    # components is like this --> [(obj, {}), (obj, {}), (obj, {})]
    def __init__(self, components):
        self.components = components
        self.obj_comp = []

    def apply_impl(self, data):
        self.obj_comp = []
        for obj, param in self.components:
            # print(param)
            aux = obj(**param)
            aux.apply_impl(data)
            self.obj_comp.append(aux)

    def use_impl(self, data):
        aux = None
        for obj in self.obj_comp:
            # print(data)
            aux = obj.use_impl(data)
        return aux

    @staticmethod
    def hps_impl(data=None):
        raise NotImplementedError("Method hps should be implemented!")
