import re

import numpy as np

from pjdata.aux.util import flatten
from pjdata.content.data import Data
from pjdata.transformer.pholder import PHolder
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.invisible import Invisible
from pjml.tool.abs.mixin.noinfo import withNoInfo


class Report(Invisible, Component):
    """Report printer.

    $r prints 'r'
    {dataset.name} prints the dataset name
    {dataset.failure} prints the failure
    """

    # TODO: 'default report' poderia ter uma sequencia de matrizes preferidas para tentar, p. ex.: ['s', 'r', 'z',
    #  ..., 'X']
    def __init__(self, text: str = "Default report r=$r", **kwargs):
        def pho(step):
            outerself = self  # TODO: verify if references to encircling class cause memory leak

            class PHo(withNoInfo, PHolder):
                def _transform_impl(self, data):
                    print(step.rjust(11), outerself._interpolate(outerself.text, data))
                    return data

            return PHo

        super().__init__(
            {"text": text}, enhancer_cls=pho("[enhancer] "), model_cls=pho("[model] "), deterministic=True, **kwargs
        )
        self.text = text

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
