import traceback

import arff
import numpy as np

from cururu.disk import save_txt
from pjml.config.cs.componentcs import ComponentCS
from pjml.config.node import Node
from pjml.config.parameter import FixedP
from pjml.tool.common.invisible import Invisible


class Save(Invisible):

    def __init__(self, name, path='./'):
        config = self._to_config(locals())
        if not path.endswith('/'):
            raise Exception('Path should end with /', path)
        if name.endswith('arff'):
            filename = path + name
        else:
            raise Exception('Unrecognized file extension:', name)
        super().__init__(config, filename, deterministic=True)
        self.model = filename
        self.filename = filename

    def _apply_impl(self, data):
        return self._use_impl(data)

    def _use_impl(self, data):
        Xt = [translate_type(typ) for typ in data.Xt]
        Yt = [translate_type(typ) for typ in data.Yt]
        dic = {
            'description': data.dataset.description,
            'relation': data.dataset.name,
            'attributes': list(zip(data.Xd, Xt)) + list(zip(data.Yd, Yt)),
            'data': np.column_stack((data.X, data.Y))
        }
        try:
            txt = arff.dumps(dic)
        except:
            traceback.print_exc()
            print('Problems creating ARFF', self.filename)
            print('Types:', Xt, Yt)
            print('Sample:', data.X[0], data.Y[0])
            print('Expected sizes:', len(Xt), '+', len(Yt))
            print('Real sizes:', len(data.X[0]), '+', len(data.Y[0].shape))
            exit(0)

        save_txt(self.filename, txt)
        return data

    @classmethod
    def _cs_impl(cls):
        params = {
            'path': FixedP('./'),
            'name': FixedP('iris.arff')
        }
        return ComponentCS(Node(params=params))


def translate_type(name):
    if isinstance(name, list):
        return name
    if name == 'real':
        return 'REAL'
    elif name == 'int':
        return 'INTEGER'
    else:
        raise Exception('Unknown type:', name)
