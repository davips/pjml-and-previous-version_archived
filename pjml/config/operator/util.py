from operator import itemgetter

from pjdata.content.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


def run(cs, train=NoData, test=NoData):
    if not isinstance(cs, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    results = []
    for pipe in cs:
        train_result, test_result = pipe.dual_transform(train, test)
        results.append((pipe, train_result, test_result))

    return results


def get(iterator, function=lambda x: (x[1].s, x[2])):
    for it in iterator:  # <-- essa ideia é tão boa que a função map() virou um padrão p/ isso [davi]
        yield function(it)


def lrun(cs, train=NoData, test=NoData):
    if not isinstance(cs, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    for pipe in cs:
        train_result, test_result = pipe.dual_transform(train, test)
        yield train_result, test_result, pipe


def sort(iterator, key=itemgetter(0), reverse=False):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""
    return sorted(iterator, key=key, reverse=reverse)


def maximize(cs, n=1, train=NoData, test=NoData):
    return min_max_pipe(cs=cs, n=n, train=train, test=test, reverse=True)


def minimize(cs, n=1, train=NoData, test=NoData):
    return min_max_pipe(cs=cs, n=n, train=train, test=test)


def min_max_pipe(cs, n=1, train=NoData, test=NoData, reverse=False):
    return ConfigList(
        components=[
            i[1] for i in sort(get(lrun(cs, train, test)), reverse=reverse)[0:n]
        ]
    )