from functools import lru_cache

from itertools import repeat

from pjdata.collection import Collection


def pca(d):
    return d * 2


def svm(d):
    return d * 1000


def rf(d):
    return d * 88


def knn(d):
    return d * 3333




def expand(data):
    generator = repeat(data)
    return Collection(generator, lambda: data)


def partition(data):
    generator = iter(range(4))
    return Collection(generator, lambda: data)


def map_(collection, f):
    generator = map(f, collection)
    return Collection(generator, lambda: collection.data)


def multi(collection, fs):
    generator = map(lambda f, x: f(x), fs, collection)
    return Collection(generator, lambda: collection.data)


def summ(collection):
    def generator():
        res = 1
        for data in collection:
            res += data
            yield data, res

    return Collection(generator(), lambda res: res)


def reduce(collection):
    # Exhaust iterator.
    for _ in collection:
        pass

    return collection.data


d = 2

d2 = reduce(
    summ(
        multi(
            partition(d),
            [svm, rf, knn, pca]
        )
    )
)
print(d2)
print()

print('-------------------------------------')
d2 = reduce(
    summ(
        multi(
            map_(
                partition(d),
                pca
            ),
            [svm, rf, knn, pca]
        )
    )
)
print(d2)

exit()


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
