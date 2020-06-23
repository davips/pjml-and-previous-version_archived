from collections import Iterable
from heapq import nsmallest, nlargest

from pjdata.aux.util import _
from pjdata.content.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


def compare(x, y):
    if isinstance(x, Iterable) and isinstance(y, Iterable):
        return all(compare(i, j) for i, j in zip(x, y))
    return x.isequal(y)


def slice(clist, start=0.75, end=1.0):
    return sort(clist, train=train, test=test,)


def run(clist, train=NoData, test=NoData):
    if not isinstance(clist, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    results = []
    for pipe in clist:
        train_result, test_result = pipe.dual_transform(train, test)
        results.append((pipe, train_result, test_result))

    return results


def lrun(clist, train=NoData, test=NoData):
    if not isinstance(clist, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    for pipe in clist:
        train_result, test_result = pipe.dual_transform(train, test)
        yield pipe, train_result, test_result


def sort(clist, train=NoData, test=NoData, key=lambda x: x[2].S, reverse=False):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""

    def iterator():
        for r in lrun(clist, train, test):
            yield key(r)

    return sorted(iterator(), key=key, reverse=reverse)


def best(clist, n=1, train=NoData, test=NoData, better="higher"):
    print(type(clist))
    if not isinstance(clist, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")
    higher = "higher"
    smaller = "smaller"

    select = None
    if better == "higher":
        select = nlargest
    elif better == "smaller":
        select = nsmallest
    else:
        raise ValueError(
            f"Expected '{higher}' or '{smaller}' in 'better', but was given '{better}'"
        )

    def dual(component):
        return component.dual_transform(train, test)[1], component

    return ConfigList(components=map(_[1], select(n, map(dual, clist))))


def maximize(clist, n=1, train=NoData, test=NoData):
    return best(clist, n=n, train=train, test=test, better="smaller")


def minimize(clist, n=1, train=NoData, test=NoData):
    return best(clist, n=n, train=train, test=test, better="higher")
