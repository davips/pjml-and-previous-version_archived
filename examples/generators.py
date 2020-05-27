from functools import lru_cache

from itertools import repeat


# Usado apenas em pipes que tenham multi.

#
# def expand(d):
#     return repeat(d)
#
#
# # Não está precisando do Expand (idem pro sample; obs. sample pode ser infinito)
# # E seria raro alguém precisar dele ser Coll -> Coll,
# # mesmo assim é bem fácil transformá-lo nisso, pondo um Reduce antes.
# def partition(d):
#     for cc in range(7):
#         yield cc
#     yield None  # None marca que o próximo é o pendurado.
#     yield d
#
#
# def map(c, f):
#     for d in c:
#         r = f(d)
#         print('map gerou', r)
#         if d is None:  # é bom checar se loop é infinito e dar hint.
#             break
#         yield r
#     yield None
#     yield next(c)
#
#     # multi vai ignorar datas excedentes!
#
#
# def multi(c, fs):
#     for f in fs:  # tratar StopException com hint sobre pipeline?
#         d = next(c)
#         if d is None:
#             raise Exception('Less Data objects than expected!')
#         r = f(d)
#         print('     multi gerou', r)
#         yield r
#     yield None
#     yield next(c)
#
#
# def summ(c):
#     res = 0
#     for d in c:
#         if d is None:
#             break
#         res *= d
#         yield d
#     yield None
#     yield res
#
#
# # Reduce saiu p/ bater c/ o Expand e pq pode não ser aplicado no prior.
# # def summ(prior_, posterior_):
# #     for d in posterior_:
# #         d += 1
# #     return prior_, posterior_
#
#
# def reduce(c):
#     for d in c:
#         if d is None:
#             break
#     return next(c)


def pca(d):
    return d * 10


def svm(d):
    return d * 1000


def rf(d):
    return d * 88


def knn(d):
    return d * 3333


class Collection:

    def __init__(self, generator, finalizer):
        self.generator = generator
        self.finalizer = finalizer
        self._last_acc = None
        self._data = None

    def __iter__(self):
        return self

    def __next__(self):
        # P/ Partition: raise StopIteration('No more Data objects left!')
        try:
            data, acc = next(self.generator)
            self._last_acc = acc
            return data, acc
        except StopIteration as e:
            self._data = self.finalizer(self._last_acc)
            # TODO: use the method bellow all over pjml.
            # HINT: To avoid nesting exceptions.
            raise e from None

    @property
    @lru_cache()
    def data(self):
        if self._data is None:
            raise ('Data object not ready!')
        return self._data


def expand(data):
    generator = repeat((data, None))
    return Collection(generator, lambda: data)


def partition(data):
    generator = zip(range(10), repeat(None))
    return Collection(generator, lambda: data)


def map_(collection, f):
    generator = zip(map(f, collection), repeat(None))
    return Collection(generator, lambda: collection.data)


def multi(collection, fs):
    generator = map(lambda f, x: (f(x), None), fs, collection)
    return Collection(generator, lambda: collection.data)


def summ(collection):
    def generator():
        res = 1
        for data in collection:
            res *= data
            yield data, res

    return Collection(generator, lambda res: res)


def reduce(c):
    for d in c:
        if d is None:
            break
    return next(c)


d = 2

print('exp mult summ red')
d2 = reduce(
    summ(
        multi(
            map(expand(d), pca),
            [svm, rf, knn]
        )
    )
)
print(d2)
print()

print('part summ red')
d2 = reduce(
    summ(
        multi(partition(d), [pca, svm, rf, knn, svm, rf, knn]),
    )
)
print(d2)

exit()
