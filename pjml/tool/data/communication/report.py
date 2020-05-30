import re

import numpy as np

from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.invisible import TInvisible
from pjml.tool.abc.mixin.transformer import TTransformer
from pjml.util import flatten


class TReport(TInvisible):
    """Report printer.

    $r prints 'r'
    {dataset.name} prints the dataset name
    {dataset.failure} prints the failure
    """

    def __init__(self, text='Default report r=$R', **kwargs):
        super().__init__({'text': text}, deterministic=True, **kwargs)
        self.text = text

    def _model_impl(self, model):
        return self._enhancer_impl('[modeler]')

    def _enhancer_impl(self, step='[enhancer]'):
        def func(posterior):
            print(step, self._interpolate(self.text, posterior))
            return posterior

        return TTransformer(
            func=func,
            info=None
        )

    @classmethod
    def _interpolate(cls, text, data):
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
    def _eval(cls, text, data):
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
    def _cs_impl(cls):
        return EmptyCS()
