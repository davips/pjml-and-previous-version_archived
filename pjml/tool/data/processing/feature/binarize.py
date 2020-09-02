from typing import Dict, Any

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from pjdata import types as t
from pjdata.creation import nominal_idxs
from pjdata.transformer.enhancer import DSStep
from pjml.config.description.cs.abc.configspace import ConfigSpace
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abs.component import Component
from pjml.tool.abs.mixin.noinfo import withNoInfo


class Binarize(Component):
    """Convert all nominal attributes to numeric by one-hot encoding."""

    def __init__(self, **kwargs):
        class Step(withNoInfo, DSStep):
            # TODO: check Data object compatibility with applied one.
            # TODO: update Xt/Xd.
            def _transform_impl(self, data: t.Data) -> t.Result:
                data_nominal_idxs = nominal_idxs(data.Xt)
                encoder = OneHotEncoder()
                matrices = {}
                if len(data_nominal_idxs) > 0:
                    nom = encoder.fit_transform(data.X[:, data_nominal_idxs]).toarray()
                    num = np.delete(data.X, data_nominal_idxs, axis=1).astype(float)
                    matrices["X"] = np.column_stack((nom, num))

                return matrices

        super().__init__({}, enhancer_cls=Step, model_cls=Step, deterministic=True, **kwargs)

    @classmethod
    def _cs_impl(cls) -> ConfigSpace:
        return EmptyCS()
