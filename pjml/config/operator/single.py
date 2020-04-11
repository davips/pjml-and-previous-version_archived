"""Operations over a single CS."""
from pjml.config.description.parameter import FixedP


def hold(cs, **kwargs):
    """Freeze args passed via kwargs. Only applicable to ComponentCS.

    Keyworded args are used to freeze some parameters of
    the algorithm, regardless of what a CS sampling could have chosen.
    TODO: it may be improved to effectively traverse and change the tree
      in-place, not just extend overwritting it
    """
    cs = cs.cs
    new_nodes = []
    for node in cs.nodes:
        params = {} if node.params is None else node.params.copy()
        for k, v in kwargs.items():
            params[k] = FixedP(v)
        new_nodes.append(node.updated(params=params))
    return cs.updated(nodes=new_nodes)


def replace(cs, **kwargs):
    """Replace parameters in CS

    It can be used to replace a FixedP by a CatP, for instance.

    Example:
        File.cs  # contents:
        # TransformerCS(Node(
        #    params={
        #       'path': FixedP('./'),
        #       'name': FixedP('iris.arff')
        #   }
        # ))

        datasets = ['iris.arff', 'car.csv', 'abalone.arff']
        cs = replace(File.cs, name=CatP(choice, items=datasets))
        cs  # contents:
        # TransformerCS(Node(
        #    params={
        #       'path': FixedP('./'),
        #       'name': CatP(choice, items=datasets)
        #   }
        # ))


    TODO: it may be improved to effectively traverse and change the tree
        in-place, not just extend overwritting it
    """
    raise NotImplementedError
