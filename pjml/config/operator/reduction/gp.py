from operator import itemgetter

from pjdata.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


def full(cs, data=NoData, n=1, field='S'):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""
    # TODO: seed?
    if not isinstance(cs, ConfigList):
        raise Exception('Exhaustive search is only possible on FiniteCS!')

    results = []
    for pipe in cs:
        p = pipe
        result_data = p.apply()
        res = p.use(result_data).field(field, context='full search').item(0)
        results.append((p, -res))

    pipes = [x[0] for x in sorted(results, key=itemgetter(1))[:n]]
    return ConfigList(components=pipes)
