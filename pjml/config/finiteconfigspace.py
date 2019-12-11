import traceback

from pjml.base.transformer import Transformer
from pjml.config.configspace import ConfigSpace
from pjml.config.distributions import choice


class FiniteConfigSpace(ConfigSpace):
    """Iterable tree representing a finite set of (hyper)parameter spaces.

    TODO: fix it to traverse more than just the top level nested nodes.
    TODO: decide if prohibition of RealP will be enforced.
    """
    current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index > len(self.nested):
            self.current_index = 0
            raise StopIteration('No more Data objects left.')
        return self.nested[self.current_index]
