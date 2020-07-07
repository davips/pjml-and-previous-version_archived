from typing import Dict, Any

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from pjdata import types as t
from pjdata.data_creation import nominal_idxs
from pjml.config.description.cs.abc.configspace import ConfigSpace
from pjml.config.description.cs.emptycs import EmptyCS
from pjml.tool.abs.component import Component


class Binarize(Component):
    """Convert all nominal attributes to numeric by one-hot encoding."""

    def __init__(self, **kwargs):
        super().__init__({}, deterministic=True, **kwargs)

    def _enhancer_info(self, data: t.Data) -> Dict[str, Any]:
        raise NotImplementedError

    def _model_info(self, data: t.Data) -> Dict[str, Any]:
        return {}

    def _enhancer_func(self) -> t.Transformation:
        # TODO: check Data object compatibility with applied one.
        # TODO: update Xt/Xd.
        def transform(data):
            data_nominal_idxs = nominal_idxs(data.Xt)
            encoder = OneHotEncoder()
            matrices = {}
            if len(data_nominal_idxs) > 0:
                nom = encoder.fit_transform(
                    data.X[:, data_nominal_idxs]
                ).toarray()
                num = np.delete(data.X,
                                data_nominal_idxs, axis=1).astype(float)
                matrices['X'] = np.column_stack((nom, num))

            return matrices

        return transform

    def _model_func(self, data: t.Data) -> t.Transformation:
        return self._enhancer_func()

    @classmethod
    def _cs_impl(cls) -> ConfigSpace:
        return EmptyCS()
