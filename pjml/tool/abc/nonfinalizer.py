import operator
from abc import ABC

from itertools import tee

from pjml.tool.abc.mixin.batch import Batch


class NonFinalizer(Batch, ABC):
    """Mixin for all Batch classes that do not accumulate results."""

# Mixins devem vir antes na MRO, se implementam ou sobrescrevem parent métodos.
# class A(ABC):
#     @abstractmethod
#     def m(self):
#         pass
#
#
# class B:
#     def m(self):
#         pass
#
#
# class C(B, A):
#     pass
