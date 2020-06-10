# Reduce saiu p/ bater c/ o Expand e pq pode não ser aplicado no prior.
# def summ(prior_, posterior_):
#     for d in posterior_:
#         d += 1
#     return prior_, posterior_


#
# from pjdata.collection import Collection
#
import itertools


def pca(d):
    return d * 2


def svm(d):
    return d * 1000


def rf(d):
    return d * 88


def knn(d):
    return d * 3333


#
#
#
# def expand(data):
#     generator = repeat(data)
#     return Collection(generator, lambda: data)
#
#
# def partition(data):
#     generator = iter(range(4))
#     return Collection(generator, lambda: data)
#
#
# def map_(collection, f):
#     generator = map(f, collection)
#     return Collection(generator, lambda: collection.data)
#
#
# def multi(collection, fs):
#     generator = map(lambda f, x: f(x), fs, collection)
#     return Collection(generator, lambda: collection.data)
#
#
# def summ(collection):
#     def generator():
#         res = 1
#         for data in collection:
#             res += data
#             yield data, res
#
#     return Collection(generator(), lambda res: res)
#
#
# def reduce(collection):
#     # Exhaust iterator.
#     for _ in collection:
#         pass
#
#     return collection.data
#
#
# d = 2
#
# d2 = reduce(
#     summ(
#         multi(
#             partition(d),
#             [svm, rf, knn, pca]
#         )
#     )
# )
# print(d2)
# print()
#
# print('-------------------------------------')
# d2 = reduce(
#     summ(
#         multi(
#             map_(
#                 partition(d),
#                 pca
#             ),
#             [svm, rf, knn, pca]
#         )
#     )
# )
# print(d2)
#
# exit()
# class StopIteration(StopIteration):
#     def __init__(self, a):
#         self.args = (a,)


def resultful(c):
    class Generator:
        def __init__(self, gen):
            self.gen = gen

        # def __next__(self):
        #     print('123 ------------------------')
        #     try:
        #         return next(self.gen)
        #     except StopIteration as e:
        #         self.result = e.args[0]
        #         return self.result

        def __iter__(self):
            self.result = yield from self.gen

    return Generator(c)


class StopIterationAndReturn(Exception):
    pass


def force_result(gen):
    try:
        gen.throw(Exception)
    except StopIteration as e:
        return e.value


def repeat(d0):
    try:
        yield from itertools.repeat(d0)
    finally:  # Infinite loop will be interrupted by throw().
        return d0


def partition(d):
    try:
        for cc in range(1, 5):
            yield cc
    finally:
        return d


def multi(c, fs):
    for f, d in zip(fs, c):
        yield f(d)
    print('>>>>>', force_result(c))
    return


gen = multi(repeat(7), [pca, svm, knn])
for d in gen:
    print(d)

# exit()
#
# gen = repeat(7)
# for d in gen:
#     print(d)
#     gen.close()
#     print(gen.value)

print('______________________________')
gen = resultful(partition(7))
for d in gen:
    print(d)
print(gen.result)
exit()


def map_(c, f):
    for d in c:
        yield f(d)
    return c.value


def multi(c, fs):
    i = 0
    watched_gen = resultful(c)
    for d in watched_gen:  # tratar StopException com hint sobre pipeline?
        print('mu nex')
        r = fs[i](d)
        i += 1
        if i == len(fs):
            break
        print('     multi gerou', r)
        yield r
    try:
        next(c)
    except StopIteration as e:
        return e.value
    else:  # caso infinito: então esse multi é o primeiro finito da cadeia
        return


def summ(c):
    res = 1
    for d in c:
        res *= d
        yield d
    return c.value.updated(res)


def reduce(c):
    for d in c:
        pass
        ...
    return c.value


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

# exit()
print('-------------------------------------')
d2 = reduce(
    summ(
        multi(
            # map_(
            repeat(d),
            # pca
            # ),
            [svm, rf, knn, pca]
        )
    )
)
print(d2)

exit()
