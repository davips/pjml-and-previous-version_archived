from sys import argv

from paje.base.data import Data
from paje.composer.Concat import Concat
from paje.evaluator.metrics import Metrics
from paje.module.modelling.classifier.nb import NB
from paje.module.noop import Noop
from paje.module.preprocessing.supervised.supmtfe import SupMtFe
from paje.module.preprocessing.unsupervised.unsupmtfe import UnsupMtFe
from paje.result.mysql import MySQL
import numpy as np


def main():
    if len(argv) < 2 or len(argv) > 5:
        print('Usage: \npython dna.py path_to_arffs arff1,arff2,arff3,... ')
    else:
        storage = MySQL()
        for a in argv:
            print(a)
        path = argv[1]
        datasets = argv[2].split(',')
        mtl = SupMtFe()  # Concat([SupMtFe()], ['X'], direction='horizontal')
        rows = []
        for dataset in datasets:
            data = Data.read_arff(path + dataset, "class")
            component = mtl.build()
            output_train, _ = storage.get_or_run(component, data,
                                                 fields_to_store=['X'])
            rows.append(output_train.X[0])

        X = np.array(rows)
        print('X ...............', X)
        noop = Noop()
        component = noop.build()
        metadata = Data(X=X, Y=[[round(x) for x in X[..., 0]]])
        storage.get_or_run(component, metadata, Data(), fields_to_store=['X'])

        nb = NB().build()
        nb.apply(metadata)
        res = nb.use(metadata)
        print('Error: ', Metrics.error(res))


if __name__ == '__main__':
    main()
