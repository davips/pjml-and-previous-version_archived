from operator import itemgetter

from pjdata.content.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


def full(cs, data=NoData, n=1, field='S'):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""
    # TODO: seed?
    if not isinstance(cs, ConfigList):
        raise Exception('Exhaustive search is only possible on FiniteCS!')

    results = []
    for pipe in cs:
        pipe.dual_transform(data, data)
        model = pipe.apply()
        res = model.use(model.data).field(field, context='full search').item(0)

        results.append((pipe, -res))

    pipes = [x[0] for x in sorted(results, key=itemgetter(1))[:n]]
    return ConfigList(components=pipes)


def sort(cs, data=NoData, n=1, field='S'):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""
    # TODO: seed?
    if not isinstance(cs, ConfigList):
        raise Exception('Exhaustive search is only possible on FiniteCS!')

    results = []
    for pipe in cs:
        pipe.dual_transform(data, data)
        model = pipe.apply()
        res = model.use(model.data).field(field, context='full search').item(0)

        results.append((pipe, -res))

    pipes = [x[0] for x in sorted(results, key=itemgetter(1))[:n]]
    return ConfigList(components=pipes)


def lrun(cs, train=NoData, test=NoData):
    # TODO: seed?
    if not isinstance(cs, ConfigList):
        raise Exception('Exhaustive search is only possible on FiniteCS!')

    for pipe in cs:
        train_result, test_result = pipe.dual_transform(train, test)
        yield pipe, train_result, test_result


def run(cs, train=NoData, test=NoData):
    # TODO: seed?
    if not isinstance(cs, ConfigList):
        raise Exception('Exhaustive search is only possible on FiniteCS!')

    results = []
    for pipe in cs:
        train_result, test_result = pipe.dual_transform(train, test)
        results.append((pipe, train_result, test_result))

    return results
