from heapq import nlargest, nsmallest

from pjdata.aux.util import _
from pjdata.content.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


# TODO: p/ consistencia de configs, elas só vão aceitar matrizes; vou ainda atualizar os componentes
#  Edesio: não sei se entendi seus objetivos, mas baseado na sua implementação condensei aqui de forma simples.
#  R: Obrigado Davi, tentei criar algumas ferramentas básicas para coletar/trabalhar com resultados e otimizar pipeline

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
        aux = component.dual_transform(train, test)[1], component
        print(aux[0])
        return aux

    select = nlargest if better == 'higher' else nsmallest
    return ConfigList(
        components=map(
            _[1],
            select(
                n,
                map(dual, clist)
            )
        )
    )
