import itertools


def pca(d):
    return d * 2


def svm(d):
    return d * 3


def rf(d):
    return d * 4


def knn(d):
    return d * 5


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


def force_result(gen):
    try:
        gen.throw(Exception)
    except StopIteration as e:
        return e.value


def repeat(d0):
    try:
        yield from itertools.repeat(d0)
    finally:  # Infinite loop will be interrupted by force_result().
        return d0  # An initial generator should ensure closing.


def partition(d):
    try:
        yield from range(1, 5)
    finally:
        return d  # An initial generator should ensure closing.


# def map(c, f):
#     try:
#         for d in c:
#             yield f(d)
#     finally:
#         return force_result(c)  # .


def multi(c, fs):
    for f, d in zip(fs, c):
        yield f(d)
    return force_result(c)  # A finite generator truncates its source.


def summ(c):
    gen = resultful(c)
    acc = 0
    for d in gen:
        acc += d
        yield d
    return acc * gen.result  # An infinite generator consumes its source until the end.


def reduce(c):
    gen = resultful(c)
    for d in gen:
        print(d)
    return gen.result  # Reduce obviously consumes its source until the end.


print(reduce(summ(map(lambda x: x * -1, partition(2)))), '\n__________________')
print(reduce(summ(multi((x for x in map(lambda x: x * -1, partition(2))), [pca, svm, rf]))), '\n__________________')
