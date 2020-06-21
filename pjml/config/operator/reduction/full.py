from heapq import nlargest, nsmallest
from operator import itemgetter

from pjdata.aux.util import _
from pjdata.content.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


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
    for it in iterator:  # <-- essa ideia é tão boa que a função map() virou um padrão p/ isso [davi]
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


# TODO: p/ consistencia de configs, elas só vão aceitar matrizes; vou ainda atualizar os componentes
#  Edesio: não sei se entendi seus objetivos, mas baseado na sua implementação condensei aqui de forma simples.
def best(clist, n=1, train=NoData, test=NoData, better='higher'):
    """"Sample" the 'n' best evaluation-pipelines.
    # <-- TODO: essa terminologia faz sentido? ou devemos encontrar outro nome pro núcleo que exclui métrica (PCA->MLP)?
    ps.1 We call the component evaluation-pipeline when it is expected to produce 's' or 'r' fields.  
    ps.2 We assume that even if training accuracy is desired, it will be already transferred to the test set as 's'/'r'.
    
    Parameters
    ----------
    clist
        Config list - it is a finite config space (CS)
    n
        number of pipelines to grab
    train
    test
    better

    Returns
    -------
        'n' best pipelines
    """
    if not isinstance(clist, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    def dual(component):
        return component.dual_transform(train, test)[1], component

    select = nlargest if better == 'higher' else nsmallest
    return ConfigList(components=map(_[1], select(n, map(dual, clist))))
