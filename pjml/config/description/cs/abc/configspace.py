from abc import abstractmethod
from functools import lru_cache

from pjdata.mixin.printable import Printable


class ConfigSpace(Printable):
    """Tree representing a (probably infinite) set of (hyper)parameter spaces.
    """
    _name = None

    def __init__(self, jsonable):
        jsonable.update(cs=self.__class__.__name__[0:-2].lower())
        super().__init__(jsonable)

    @abstractmethod
    def sample(self):
        pass

    @property
    def cs(self):
        """Shortcut to ease retrieving a CS from a Transformer class without
        having to check that it is not already a CS."""
        return self

    # Amenities.
    @property
    @lru_cache()
    def name(self):
        if self._name is None:
            self._name = self.__class__.__name__[0:-2].lower()
        return self._name

    @property
    @lru_cache()
    def longname(self):
        long = ''
        for things in ['transformers', 'components']:
            if things in self.jsonable:
                items = ', '.join(tr.longname for tr in self.jsonable[things])
                long = f'[{items}]'
        return self.name + long
