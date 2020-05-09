from abc import ABC
from functools import lru_cache

from pjdata.aux.decorator import classproperty
from pjml.tool.abc.transformer import DTransformer


class Container(DTransformer, ABC):
    """A container modifies 'transformer(s)'."""

    def __init__(self, config, seed, transformers, deterministic):
        if not transformers:
            raise Exception(
                f'A container ({self.name}) should have at least one '
                f'transformer!')

        # transformers=[Chain(A)] should appear as transformers=[A] in config.
        from pjml.tool.chain import Chain
        if len(transformers) == 1 and isinstance(transformers[0], Chain):
            transformers = transformers[0].transformers

        # Propagate seed.
        self.transformers = []
        for transformer in transformers:
            if not ('seed' in transformer.config or transformer.deterministic):
                transformer = transformer.updated(seed=seed)
            self.transformers.append(transformer)

        complete_config = {'transformers': self.transformers}
        complete_config.update(config)
        super().__init__(complete_config,
                         deterministic=deterministic,
                         nodata_handler=self.transformers[0].nodata_handler)

    @property
    @lru_cache()
    def wrapped(self):
        from pjml.tool.meta.wrap import Wrap
        for transformer in self.transformers:
            transformer = transformer.wrapped
            if isinstance(transformer, Wrap):
                return transformer
        return None

    @classmethod
    @classproperty
    def cs(cls):
        raise Exception(
            f'{cls.name} depends on transformers to build a CS.\n'
            f'Just instantiate the class {cls.name} instead of calling its .cs!'
        )

    @property
    @lru_cache()
    def longname(self):
        names = ", ".join([tr.longname for tr in self.transformers])
        return self.name + f'[{names}]'

    @classmethod
    def _cs_impl(cls):
        raise Exception(f'Wrong calling of {cls.name}._cs_impl!')

    def __str__(self, depth=''):
        if not self.pretty_printing:
            return super().__str__()

        inner = []
        for t in self.transformers:
            inner.append('    ' + t.__str__(depth).replace('\n', '\n' + '    '))

        return f'{depth}{self.name}>>\n' + \
               '\n'.join(inner) + \
               f'\n{depth}<<{self.name}'
