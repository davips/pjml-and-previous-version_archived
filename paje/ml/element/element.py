import inspect
from abc import ABC

from paje.base.component import Component
from paje.util.misc import unsandwich


class Element(Component, ABC):

    def modifies(self, op):
        if op not in ['a', 'u']:
            raise Exception('Wrong op:', op)

        if self._modified[op] is None:
            from paje.base.data import Data
            func = self.apply_impl if op == 'a' else self.use_impl
            source = inspect.getsource(func)

            # Handle the case where use() is called when applying().
            if op == 'a' and 'self.use_impl' in source:
                self._modified[op] = self.modifies('u')
            else:
                self._modified[op] = []

            lines = source.split('\n')
            for line in lines:
                if 'updated' in line:
                    meat = unsandwich(line)
                    args = [arg for arg in meat.split(',') if '=' in arg]
                    fields = [arg.split('=')[0].strip() for arg in args]
                    for f in fields:
                        if len(f) == 1:
                            self._modified[op].append(
                                Data.fields_in_lowercase_format[f]
                            )
            self._modified[op] = list(set(self._modified[op]))
        return self._modified[op]
