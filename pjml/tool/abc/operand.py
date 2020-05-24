from abc import ABCMeta


class FlatSelect:
    def __init__(self, *operands):
        self.operands = operands


class MetaOperand(ABCMeta):
    def __add__(self, other):
        from pjml.config.operator.many import select
        if isinstance(other, FlatSelect):
            return FlatSelect(*other.operands)
        print(self, other)
        return select(self, other)


class Operand(metaclass=MetaOperand):
    def __add__(self, other):
        from pjml.tool.chain import Chain
        if isinstance(other, FlatSelect):
            return FlatSelect(*other.operands)
        return Chain(self, other)
