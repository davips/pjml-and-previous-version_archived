import re

import numpy as np

import pjdata.types as t
from pjdata.aux.util import flatten
from pjdata.content.data import Data
from pjdata.transformer.pholder import PHolder
from pjdata.transformer.transformer import Transformer
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.invisible import Invisible


class Report(Invisible, Component):
    """Report printer.

    $r prints 'r'
    {dataset.name} prints the dataset name
    {dataset.failure} prints the failure
    """

    # TODO: 'default report' poderia ter uma sequencia de matrizes preferidas para tentar, p. ex.: ['s', 'r', 'z',
    #  ..., 'X']
    def __init__(self, text: str = "Default report r=$r", **kwargs):
        super().__init__({"text": text}, deterministic=True, **kwargs)
        self.text = text

    def _enhancer_impl(self) -> Transformer:
        return self._pholder("[enhancer] ")

    def _model_impl(self, data: t.Data) -> Transformer:
        return self._pholder("[model] ")

    def _pholder(self, step) -> PHolder:
        def transform(data):
            print(step.rjust(11), self._interpolate(self.text, data))
            return data

        return PHolder(self, transform)

    @classmethod
    def _interpolate(cls, text: str, data: Data) -> str:
        # TODO: global(?) option to re-prettify line breaks from numpy arrays
        def samerow(M):
            return np.array_repr(M).replace("\n      ", "").replace("  ", "")

        def f(obj_match):
            field = obj_match.group(1)
            M = data.field(field, context=cls)
            if isinstance(M, np.float64):
                return str(M)
            try:
                if np.issubdtype(M, np.number):
                    return samerow(np.round(M, decimals=4))
            finally:
                return samerow(M)

        p = re.compile(r"\$([~a-zA-Z]+)")
        return cls._eval(p.sub(f, text), data)

    @classmethod
    def _eval(cls, text: str, data: Data) -> str:
        txt = ""
        run = False
        expanded = [w.split("}") for w in ("_" + text + "_").split("{")]
        for seg in flatten(expanded):
            if run:
                code = "data." + seg
                try:
                    # Convert mapping through ~ operand.
                    if "~" in seg:
                        iterable, accessor = seg.split("~")
                        code = f"[item.{accessor} for item in data.{iterable}]"
                    if "^" in seg:
                        iterable, accessor = seg.split("^")
                        code = f"data.{iterable} ^ '{accessor}'"

                    # Data cannot be changed, so we don't use exec, which would accept dangerous assignments.
                    txt += str(eval(code))
                except Exception as e:
                    print(f"Problems parsing\n  {text}\nwith data\n  {data}\n" f"{data.history}\n!")
                    raise e
            else:
                txt += seg
            run = not run
        return txt[1:][:-1]

    @classmethod
    def _cs_impl(cls) -> EmptyCS:
        return EmptyCS()
