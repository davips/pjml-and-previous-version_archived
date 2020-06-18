from abc import ABC

from pjml.config.description.cs.abc.configspace import ConfigSpace


class OperatorCS(ConfigSpace, ABC):

    def __init__(self, *components):
        print("create a OperatorCS")
        components = [compo.cs for compo in components]
        print("components_cs: ", components)
        super().__init__({'components': components})
        self.components = components
