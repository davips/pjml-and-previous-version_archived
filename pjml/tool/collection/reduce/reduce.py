from functools import lru_cache
from typing import Tuple, Any, Dict

import pjdata.types as t
from pjdata.content.data import Data
from pjdata.transformer.enhancer import Enhancer
from pjdata.transformer.model import Model
from pjml.config.description.cs.cs import CS
from pjml.config.description.node import Node
from pjml.tool.abs.component import Component
from pjml.tool.abs.invisible import Invisible


class Reduce(Invisible, Component):
    def __init__(self, config: dict = None, **kwargs):
        # TODO: delete onenhance/onmodel? senão, consumir pode explodir
        config = {} if config is None else config
        super().__init__(config, deterministic=True, **kwargs)

    @lru_cache()
    def _enhancer_info(self, data: t.Data = None) -> Dict[str, Any]:
        return {}

    @lru_cache()
    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> t.Transformation:
        def transform(data: Data) -> t.Result:
            # Exhaust iterator.
            frozen, failure = False, None
            for d in data.stream:
                frozen = frozen or d.isfrozen
                failure = failure or d.failure  # TODO: is it ok to just get the last failure?
            return {'stream': None, 'frozen': frozen, 'failure': failure}

        return transform

    def _model_func(self, data: t.Data) -> t.Transformation:
        return self._enhancer_func()

    @classmethod
    def _cs_impl(cls):
        params = {}
        return CS(nodes=[Node(params)])

    def dual_transform(self, train: t.Data, test: t.Data) -> Tuple[t.Data, t.Data]:
        if self.hasenhancer and self.hasmodel:
            afrozen, afailure = False, None
            bfrozen, bfailure = False, None
            for a, b in zip(train.stream, test.stream):
                afrozen = afrozen or a.isfrozen
                bfrozen = bfrozen or b.isfrozen
                afailure = afailure or a.failure  # TODO: is it ok to just get the last failure?
                bfailure = bfailure or b.failure  # TODO: is it ok to just get the last failure?
            train = train.transformedby(
                Enhancer(self, lambda _: {'stream': None, 'frozen': afrozen, 'failure': afailure}, lambda: {})
            )
            test = test.transformedby(
                Model(self, lambda _: {'stream': None, 'frozen': bfrozen, 'failure': bfailure}, {}, train)
            )
        elif self.hasenhancer:
            train = self.enhancer.transform(train)
        elif self.hasmodel:
            test = self.model(train).transform(test)
        return train, test
