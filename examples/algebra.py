from pjml.tool.data.processing.feature.reductor.pca import PCA

cs = (PCA @ PCA) * PCA()
print(cs.longname)

print()
p = PCA() * PCA() * PCA() * PCA()
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
