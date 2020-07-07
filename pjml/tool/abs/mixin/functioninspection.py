from functools import lru_cache


class withFunctionInspection:

    @classmethod
    @lru_cache()
    def function_from_name(cls):
        """Map each function name to its corresponding class method."""
        return {name: getattr(cls, "_fun_" + name) for name in cls.names()}

    @classmethod
    @lru_cache()
    def names(cls):
        return [name.split("_fun_")[1] for name in dir(cls) if "_fun_" in name]
