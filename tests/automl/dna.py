from sys import argv

from paje.base.data import Data
from paje.result.mysql import MySQL


def main():
    if len(argv) < 2 or len(argv) > 5:
        print('Usage: \npython toy.py path_to_arffs arff1,arff2,arff3,... ')
    else:
        storage = MySQL()
        for a in argv:
            print(a)
        path = argv[1]
        datasets = argv[2].split(',')
        # mtl = HConcat(SupMtL(), UnsupMtL())
        for dataset in datasets:
            data = Data.read_arff(path + dataset, "class")
            output_train, _ = storage.get_or_run(mtl, data, Data(),
                                                      fields_to_store=['X'])


if __name__ == '__main__':
    main()
