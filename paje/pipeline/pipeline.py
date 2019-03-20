from paje.base.component import Component
from paje.base.hps import HPTree

class Pipeline(Component):

    # components is like this --> [(obj, {}), (obj, {}), (obj, {})]
    def __init__(self, components):
        self.components = components
        self.obj_comp = []

    def apply(self, data):
        self.obj_comp = []
        for obj, param in self.components:
            # print(param)
            aux = obj(**param)
            aux.apply(data)
            self.obj_comp.append(aux)

    def use(self, data):
        aux = None
        for obj in self.obj_comp:
            # print(data)
            aux = obj.use(data)
        return aux

    @staticmethod
    def hps_impl(data=None):
        raise NotImplementedError("Method hps should be implement!")
