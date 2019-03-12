class Pipeline():

    # components is like this --> [(obj, {}), (obj, {}), (obj, {})]
    def __init__(self, components):
        self.components = components
        self.obj_comp = []

    def apply(self, data):
        aux = None
        self.obj_comp = []
        for obj, param in self.components:
            aux = obj(*param).apply(data)
        return aux

    def use(self, data):
        aux = None
        for obj in self.obj_comp:
            aux = obj.use(data)
        return aux

    def set_param(self, **kwargs):
        pass
