from pjml.config.description.cs.abc.configspace import ConfigSpace
from pjml.config.description.distributions import choice


class ConfigList(ConfigSpace):
    """Traversable discrete finite CS.

    Iterable CS. This CS does not accept config spaces, only transformers.

    components
        A list of transformers.
    """

    def __init__(self, *args, transformers=None):
        if transformers is None:
            transformers = args
        super().__init__({'transformers': transformers})

        from pjml.tool.abc.mixin.component import Component
        for transformer in transformers:
            if not (isinstance(transformer, Component)):
                raise Exception(
                    f'\nGiven: {type(transformer)}\n{transformer}\n'
                    f'ConfigList does not accept config spaces, '
                    f'only components!')
        self.current_index = -1
        self.size = len(transformers)
        self.transformers = transformers
        self._name = 'list'

    def sample(self):
        return choice(self.transformers)

    def __iter__(self):
        return self.transformers.__iter__()

    # def __next__(self):
    #     self.current_index += 1
    #     if self.current_index >= self.size:
    #         self.current_index = -1
    #         raise StopIteration('No more objects left.')
    #     return self.transformers[self.current_index]
    #
    # def __len__(self):
    #     return self.size
    #
    # def __getitem__(self, idx):
    #     return self.transformers[idx]
