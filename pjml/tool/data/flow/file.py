from functools import lru_cache
from typing import List

from pjdata.data_creation import read_arff
from pjdata.specialdata import NoData
from pjdata.step.transformation import Transformation
from pjdata.step.transformer import Transformer
from pjml.config.description.cs.transformercs import TransformerCS
from pjml.config.description.node import Node
from pjml.config.description.parameter import FixedP
from pjml.tool.abc.mixin.component import Component
from pjml.tool.abc.mixin.nodatahandler import NoDataHandler

# Precisa herdar de Invisible, pois o mesmo Data pode vir de diferentes
# caminhos de arquivo (File) ou servidores (Source) e essas informações são
# irrelevantes para reprodutibilidade. Herdando de Invisible, o histórico é [].


class File(Component, NoDataHandler):
    """Source of Data object from CSV, ARFF, file.

    TODO: always classification task?
    There will be a single transformation (history) on the generated Data.

    A short hash will be added to the name, to ensure unique names.
    Actually, the first collision is expected after 12M different datasets
    with the same name ( 2**(log(107**7, 2)/2) ).
    Since we already expect unique names like 'iris', and any transformed
    dataset is expected to enter the system through a transformer,
    12M should be safe enough. Ideally, a single 'iris' be will stored.
    In practice, no more than a dozen are expected.
    """

    def __init__(
            self,
            name: str,
            path: str = './',
            description: str = 'No description.',
            hashes: str = None,
            **kwargs
    ):

        # Some checking.
        if not path.endswith('/'):
            raise Exception('Path should end with /', path)
        if name.endswith('arff'):
            actual_hashes, data = read_arff(path + name, description)
        else:
            raise Exception('Unrecognized file extension:', name)
        if hashes:
            if hashes != actual_hashes:
                raise Exception(
                    f'Provided hashes f{hashes} differs from actual hashes '
                    f'{actual_hashes}!')

        # Unique config for this file.
        config = {
            'name': name,
            'path': path,
            'description': description,
            'hashes': actual_hashes
        }
        # self._digest = md5digest(serialize(actual_hashes).encode())

        super().__init__(config, deterministic=True, **kwargs)
        self.data = data

    def _transformer(self) -> Transformer:
        def transform(posterior):  # old use/apply
            self._enforce_nodata(posterior, 'u')  # fixei 'u'
            return self.data

        return Transformer(func=transform, info=None)

    def _model_impl(self, prior: NoData) -> Transformer:
        self._enforce_nodata(prior, 'a')  # fixei 'a'
        return self._transformer()

    def _enhancer_impl(self) -> Transformer:
        return self._transformer()

    @classmethod
    def _cs_impl(cls) -> TransformerCS:
        params = {
            'path': FixedP('./'),
            'name': FixedP('iris.arff'),
            'description': FixedP('No description.'),
            'matrices_hash': FixedP('1234567890123456789')
        }

        # TODO: I think that we should set as follow:
        # TransformerCS(nodes=[Node(params=params)])
        return TransformerCS(Node(params=params))

    @lru_cache()
    def transformations(
            self,
            step: str,
            clean: bool = True
    ) -> List[Transformation]:
        return [Transformation(self, 'u')]
