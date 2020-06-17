import re
from typing import Callable, Dict, Any

import numpy as np

import pjdata.types as t
from pjdata.aux.util import flatten
from pjdata.content.data import Data
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abc.invisible import Invisible
from pjml.tool.abc.mixin.component import Component


class Report(Invisible, Component):
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

    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> Callable[[Data], Data]:
        step = '[model] '
        return lambda train: self._transform(train, step)

    def _model_info(self, data: Data) -> Dict[str, Any]:
        return {}

    def _model_func(self, train: t.Data) -> Callable[[t.Data], t.Data]:
        step = '[enhancer] '
        return lambda test: self._transform(test, step)

    def _transform(self, data: Data, step) -> Data:
        print(step, self._interpolate(self.text, data))
        return data

    @classmethod
    def _interpolate(cls, text: str, data: Data) -> str:
        # TODO: global(?) option to reprettify line breaks from numpy arrays
        def samerow(M):
            return np.array_repr(M).replace('\n      ', '').replace('  ', '')

        def f(obj_match):
            field = obj_match.group(1)
            M = data.field(field, context=cls)
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
