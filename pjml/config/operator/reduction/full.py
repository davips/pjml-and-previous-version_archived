from operator import itemgetter

from pjdata.content.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


def full(cs, data=NoData, n=1, field="S"):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""
    # TODO: seed?
    if not isinstance(cs, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    results = []
    for pipe in cs:
        pipe.dual_transform(data, data)
        model = pipe.apply()
        res = model.use(model.data).field(field, context="full search").item(0)

        results.append((pipe, -res))

    pipes = [x[0] for x in sorted(results, key=itemgetter(1))[:n]]
    return ConfigList(components=pipes)


def min_max_pipe(cs, n=1, train=NoData, test=NoData, reverse=False):
    return ConfigList(
        components=[
            i[1] for i in sort(get(lrun(cs, train, test)), reverse=reverse)[0:n]
        ]
    )


def minimize(cs, n=1, train=NoData, test=NoData):
    return min_max_pipe(cs=cs, n=n, train=train, test=test)


def maximize(cs, n=1, train=NoData, test=NoData):
    return min_max_pipe(cs=cs, n=n, train=train, test=test, reverse=True)


def sort(iterator, key=itemgetter(0), reverse=False):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""
    return sorted(iterator, key=key, reverse=reverse)


def get(iterator, function=lambda x: (x[1].s, x[2])):
    for it in iterator:
        yield function(it)


def lrun(cs, train=NoData, test=NoData):
    if not isinstance(cs, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    for pipe in cs:
        train_result, test_result = pipe.dual_transform(train, test)
        yield train_result, test_result, pipe


def run(cs, train=NoData, test=NoData):
    if not isinstance(cs, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    results = []
    for pipe in cs:
        train_result, test_result = pipe.dual_transform(train, test)
        results.append((pipe, train_result, test_result))

    return results
