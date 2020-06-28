from bisect import bisect
from operator import itemgetter

import numpy as np

from pjml.tool.abs.configless import ConfigLess


class Eq(ConfigLess):
    """Uniformly distribute examples along each attribute to make them
    independent of scale and unit measure.

    Each attribute value is replaced by the order in which the example is
    ranked according with that attribute.
    Applying a normalization after this transformer is recommended."""

    def _apply_impl(self, data):
        newX = []
        for xs in np.transpose(data.X):
            xso = sorted(enumerate(xs), key=itemgetter(1))
            xs2 = sorted(self._enumerate(xso), key=itemgetter(1))
            newxs = [x[0] for x in xs2]
            newX.append(newxs)
        np.transpose(newX)

    def _use_impl(self, data, **kwargs):
        pass

    # def _convert(self, x):
    #     bisect(self.mo)  # precisa remover repetidos, repartir o espaço original

    def _enumerate(self, lst):
        """Enumerate a sorted list, repeating the index for duplicate values."""
        idx = 0
        old = lst[0]
        res = []
        for x in lst:
            if x > old:
                idx += 1
            old = x
            res.append((idx, x))
        return res
