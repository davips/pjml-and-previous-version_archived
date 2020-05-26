from itertools import repeat


# Usado apenas em pipes que tenham multi.


def expand(d):
    return repeat(d)


# Não está precisando do Expand (idem pro sample; obs. sample pode ser infinito)
# E seria raro alguém precisar dele ser Coll -> Coll,
# mesmo assim é bem fácil transformá-lo nisso, pondo um Reduce antes.
def partition(d):
    for cc in range(7):
        yield cc
    yield None  # None marca que o próximo é o pendurado.
    yield d


def map(c, f):
    for d in c:
        r = f(d)
        print('map gerou', r)
        if d is None:  # é bom checar se loop é infinito e dar hint.
            break
        yield r
    yield None
    yield next(c)

    # multi vai ignorar datas excedentes!


def multi(c, fs):
    for f in fs:  # tratar StopException com hint sobre pipeline?
        d = next(c)
        if d is None:
            raise Exception('Less Data objects than expected!')
        r = f(d)
        print('     multi gerou', r)
        yield r
    yield None
    yield next(c)


def summ(c):
    res = 0
    for d in c:
        if d is None:
            break
        res *= d
        yield d
    yield None
    yield res


# Reduce saiu p/ bater c/ o Expand e pq pode não ser aplicado no prior.
# def summ(prior_, posterior_):
#     for d in posterior_:
#         d += 1
#     return prior_, posterior_


def reduce(c):
    for d in c:
        if d is None:
            break
    return next(c)


def pca(d):
    return d * 10


def svm(d):
    return d * 1000


def rf(d):
    return d * 88


def knn(d):
    return d * 3333


d = 2

print('Com reduce(summ(')
d2 = reduce(summ(
    multi(
        map(expand(d), pca),
        [svm, rf, knn]
    )
))
print(d2)
print()

print('partition')
d2 = reduce(summ(
    multi(partition(d), [pca, svm, rf, knn, svm, rf, knn]),
))
print(d2)

exit()
