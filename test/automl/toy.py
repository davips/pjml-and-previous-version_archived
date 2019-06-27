import sys

from paje.base.data import Data
from paje.automl.random import RandomAutoML
from paje.evaluator.metrics import Metrics
from paje.ml.element.modelling.classifier.dt import DT


def main():
    if len(sys.argv[1:]) < 1 or any(['=' not in k for k in sys.argv[1:]]):
        print('Usage: \npython toy.py data=/tmp/dataset.arff '
              '[iter=#] [seed=#] [storage=mysql/sqlite/cached] [db=dna] ')
    else:
        arg = {tupl.split('=')[0]: tupl.split('=')[1] for tupl in sys.argv[1:]}
        my_modelers = [DT()]

        for k, v in arg.items():
            print(f'{k}={v}')

        if 'storage' in arg:
            if arg['storage'] == 'sqlite':
                from paje.result.sqlite import SQLite
                storage = SQLite(debug=not True)
            elif arg['storage'] == 'mysql':
                from paje.result.mysql import MySQL
                storage = MySQL(db=arg['db'], debug=not True)
            elif arg['storage'] == 'cached':
                from paje.result.mysql import MySQL
                from paje.result.sqlite import SQLite
                storage = MySQL(db=arg['db'], nested_storage=SQLite())
            elif arg['storage'] == 'sync':
                from paje.result.mysql import MySQL
                from paje.result.sqlite import SQLite
                storage = MySQL(db=arg['db'], nested_storage=SQLite(),
                                sync=True)
            else:
                raise Exception('Wrong storage', arg['storage'])
        else:
            storage = None

        iterations = int(arg['iter']) if 'iter' in arg else 3
        random_state = int(arg['seed']) if 'seed' in arg else 0
        data = Data.read_arff(arg['data'], "class")

        trainset, testset = data.split()

        automl_rs = RandomAutoML(
            # preprocessors=default_preprocessors,
            # modelers=default_modelers,
            preprocessors=[],
            modelers=my_modelers,
            storage_for_components=storage,
            show_warns=False,
        ).build(
            max_iter=iterations,
            pipe_length=15, repetitions=2,
            random_state=random_state
        )
        automl_rs.apply(trainset)
        testout = automl_rs.use(testset)
        if testout is None:
            print('No working pipeline found!')
            exit(0)
        print("Accuracy score", Metrics.accuracy(testout))
        print()


if __name__ == '__main__':
    main()
