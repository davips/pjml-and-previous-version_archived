from pjml.tool.data.modeling.supervised.classifier.rf import RF
from pjml.tool.data.processing.instance.sampler.over.random import OverS
from pjml.tool.data.processing.instance.sampler.under.random import UnderS

cs = (UnderS @ OverS) * RF()
print(cs.longname)

print()
p = UnderS() * RF() * UnderS() * RF()
print(p.longname)

# class A:
#     def __init__(self, n):
#         self.n = n
#
#     def __mul__(self, other):
#         print(self.n, other.n)
#
#     def __rmul__(self, other):
#         print(other.n, self.n)
#
#
# a = A('a')
# b = A('b')
# a * b
# b * a
