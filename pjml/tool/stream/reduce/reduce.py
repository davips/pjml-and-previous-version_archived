from functools import lru_cache
from typing import Tuple, Any, Dict

import pjdata.types as t
from pjdata.content.data import Data
from pjdata.transformer.enhancer import Enhancer, DSStep
from pjdata.transformer.model import Model
from pjdata.transformer.pholder import PHolder
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.tool.abs.component import Component
from pjml.tool.abs.invisible import Invisible
from pjml.tool.abs.mixin.noinfo import withNoInfo


class Reduce(Invisible, Component):
    def __init__(self, config: dict = None, **kwargs):
        # TODO: delete onenhance/onmodel? senão, consumir pode explodir
        config = {} if config is None else config

        class Step(withNoInfo, DSStep):
            def _transform_impl(self, data: t.Data) -> t.Result:
                # Exhaust iterator.
                frozen, failure = False, None
                for d in data.stream:
                    frozen = frozen or d.isfrozen
                    failure = failure or d.failure  # TODO: is it ok to just get the last failure?
                return {"stream": None, "frozen": frozen, "failure": failure}

        super().__init__(config, enhancer_cls=Step, model_cls=Step, deterministic=True, **kwargs)

    @classmethod
    def _cs_impl(cls):
        params = {}
        return CS(nodes=[Node(params)])

    def dual_transform(self, train: t.Data, test: t.DataOrTup) -> Tuple[t.Data, t.DataOrTup]:
        if self.hasenhancer and self.hasmodel:
            afrozen, afailure = False, None
            bfrozen, bfailure = False, None
            for a, b in zip(train.stream, test.stream):
                afrozen = afrozen or a.isfrozen
                bfrozen = bfrozen or b.isfrozen
                afailure = afailure or a.failure  # TODO: is it ok to just get the last failure?
                bfailure = bfailure or b.failure  # TODO: is it ok to just get the last failure?
            train = train.transformedby(
                Enhancer(self, lambda _: {"stream": None, "frozen": afrozen, "failure": afailure}, lambda: {})
            )
            test = test.transformedby(
                Model(self, lambda _: {"stream": None, "frozen": bfrozen, "failure": bfailure}, {}, train)
            )
        elif self.hasenhancer:
            train = self.enhancer.transform(train)
            test = test.transformedby(PHolder(self))
        elif self.hasmodel:
            train = train.transformedby(PHolder(self))
            test = self.model(train).transform(test)
        else:
            train = train.transformedby(PHolder(self))
            test = test.transformedby(PHolder(self))

        return train, test
