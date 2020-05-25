from abc import abstractmethod, ABC

from pjdata.aux.serialization import materialize
from pjml.config.description.cs.abc.configspace import ConfigSpace
from pjml.config.description.distributions import choice
from pjml.config.description.node import Node


class ComponentCS(ConfigSpace, ABC):
    def __init__(self, name, path, components, nodes):
        if nodes is None:
            nodes = []
        jsonable = {'component': {'name': name, 'path': path}, 'nodes': nodes}
        if components:
            components = [compo.cs for compo in components]
            jsonable['components'] = components
        super().__init__(jsonable)
        for cs in nodes:
            if not isinstance(cs, Node):
                raise Exception(
                    f'{self.__class__.__name__} can only have Node as nodes.'
                    f' Not {type(cs)} !'
                )
        self._name, self.path, self.nodes = name, path, nodes
        self.components = components

    @abstractmethod
    def _sample_cfg(self):
        pass

    def sample(self):
        """Sample a completely configured transformer.

        Choose a path from tree and set values to parameters according to
        the given sampling functions.

        Returns
        -------
        A transformer
        """
        config = self._sample_cfg()

        # Fill config with values from internal nodes.
        if self.nodes:
            child_node = choice(self.nodes)
            config.update(child_node.partial_sample())

        return materialize(self.name, self.path, config)

    @abstractmethod
    def updated(self, nodes):
        pass
