from sys import argv

from paje.base.data import Data
from paje.composer.Concat import Concat
from paje.module.noop import Noop
from paje.module.preprocessing.supervised.supmtfe import SupMtFe
from paje.module.preprocessing.unsupervised.unsupmtfe import UnsupMtFe
from paje.result.mysql import MySQL


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
            output_train, _ = storage.get_or_run(component, data, data,
                                                 fields_to_store=['X'])
            print(output_train.X[0], '< row')
            rows.append(output_train.X[0])
        noop = Noop()
        component = noop.build()
        metadata = Data(X=rows)
        storage.get_or_run(component, metadata, Data(), fields_to_store=['X'])


if __name__ == '__main__':
    main()
