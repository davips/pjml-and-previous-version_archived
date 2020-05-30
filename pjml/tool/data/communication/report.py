import re

import numpy as np

from pjdata.data import Data
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.invisible import Invisible
from pjml.tool.abc.mixin.transformer import Transformer
from pjml.util import flatten


class Report(Invisible):
    """Report printer.

    $r prints 'r'
    {dataset.name} prints the dataset name
    {dataset.failure} prints the failure
    """

    def __init__(
            self,
            text: str = 'Default report r=$R',
            **kwargs
    ):
        super().__init__({'text': text}, deterministic=True, **kwargs)
        self.text = text

    def _model_impl(self, data: Data) -> Transformer:
        return self._enhancer_impl('[modeler]')

    def _enhancer_impl(
            self,
            step: str = '[enhancer]'
    ) -> Transformer:
        def func(posterior: Data) -> Data:
            print(step, self._interpolate(self.text, posterior))
            return posterior

        return Transformer(
            func=func,
            info=None
        )

    @classmethod
    def _interpolate(cls, text: str, data: Data) -> str:
        # TODO: global(?) option to reprettify line breaks from numpy arrays
        def samerow(M):
            return np.array_repr(M).replace('\n      ', '').replace('  ', '')

        def f(obj_match):
            field = obj_match.group(1)
            M = data.field(field, cls)
            try:
                if np.issubdtype(M, np.number):
                    return samerow(np.round(M, decimals=4))
            finally:
                return samerow(M)

        p = re.compile(r'\$([a-zA-Z]+)')
        return cls._eval(p.sub(f, text), data)

    @classmethod
    def _eval(cls, text: str, data: Data) -> str:
        txt = ''
        run = False
        expanded = [w.split('}') for w in ('_' + text + '_').split('{')]
        for seg in flatten(expanded):
            if run:
                try:
                    txt += str(eval('data.' + seg))
                except Exception as e:
                    print(
                        f'Problems parsing\n  {text}\nwith data\n  {data}\n'
                        f'{data.history}\n!')
                    raise e
            else:
                txt += seg
            run = not run
        return txt[1:][:-1]

    @classmethod
    def _cs_impl(cls) -> EmptyCS:
        return EmptyCS()
