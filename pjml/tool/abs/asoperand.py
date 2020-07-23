from abc import ABCMeta


class MetaOperand(ABCMeta):
    def __add__(cls, other):
        return cls.__add__(cls, other)

    def __mul__(cls, other):
        return cls.__mul__(cls, other)

    def __matmul__(cls, other):
        return cls.__matmul__(cls, other)

    # Ensures resulting object will also accept operators.
    def __radd__(self, other):
        return self.__add__(other, self)

    def __rmul__(self, other):
        return self.__mul__(other, self)

    def __rmatmul__(self, other):
        return self.__matmul__(other, self)

    # I had to put it here, since I cannot create a specific metaclass for that.
    # Such attempt would lead to the diamond inheritance problem.
    def _name_impl(cls):
        return cls.__name__


class AsOperand(metaclass=MetaOperand):
    def __add__(self, other):
        from pjml.config.description.cs.selectcs import SelectCS

        if isinstance(other, SelectCS):
            return SelectCS(self, *other.components)
        elif isinstance(self, SelectCS):
            return SelectCS(*self.components, other)
        return SelectCS(self, other)

    def __mul__(self, other):
        from pjml.tool.chain import Chain, ChainCS

        if isinstance(other, (Chain, ChainCS)):
            return Chain(self, *other.components)
        if isinstance(self, (Chain, ChainCS)):
            return Chain(*self.components, other)
        return Chain(self, other)

    def __matmul__(self, other):  # @
        from pjml.config.description.cs.shufflecs import ShuffleCS

        if isinstance(other, ShuffleCS):
            return ShuffleCS(self, *other.components)
        elif isinstance(self, ShuffleCS):
            return ShuffleCS(*self.components, other)
        return ShuffleCS(self, other)

    # Ensures resulting object will also accept operators.
    def __radd__(self, other):
        return self.__class__.__add__(other, self)

    def __rmul__(self, other):
        return self.__class__.__mul__(other, self)

    def __rmatmul__(self, other):
        return self.__class__.__matmul__(other, self)
