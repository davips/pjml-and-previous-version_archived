from operator import itemgetter

from numpy import Inf

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
        # try: não recomendado capturar exceções aqui, aumente ExceptionHandler!
        model = pipe.apply()
        res = model.use(model.data).field(field, component='full search').item(0)
        # except Exception as e:
        #     res = -Inf

        results.append((pipe, -res))

    pipes = [x[0] for x in sorted(results, key=itemgetter(1))[:n]]
    return ConfigList(transformers=pipes)
