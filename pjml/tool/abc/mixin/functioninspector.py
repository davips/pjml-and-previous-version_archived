from functools import lru_cache

from pjdata.aux.util import Property


class FunctionInspector:
    @Property
    @lru_cache()
    def function_from_name(self):
        """Map each function name to its corresponding class method."""
        return {name: getattr(self, '_fun_' + name)
                for name in self.names()}

    @classmethod
    @lru_cache()
    def names(cls):
        return [name.split('_fun_')[1] for name in dir(cls) if '_fun_' in name]
