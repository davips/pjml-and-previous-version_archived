import sys

from paje.automl.composer.pipeline import Pipeline
from paje.automl.composer.switch import Switch
from paje.automl.optimization.blind.random import RandomAutoML
from paje.base.data import Data
from paje.ml.element.modelling.supervised.classifier.dt import DT
from paje.ml.element.modelling.supervised.classifier.nb import NB
from paje.ml.element.preprocessing.unsupervised.feature.scaler.equalization import \
    Equalization
from paje.ml.metric.supervised.classification.mclassif import Metrics


def main():
    if len(sys.argv[1:]) < 1 or any(['=' not in k for k in sys.argv[1:]]):
        print('Usage: \npython toy.py data=/tmp/dataset.arff '
              '[iter=#] [seed=#] [storage=mysql/sqlite/cached/dump] ['
              'db=dna] ')
    else:
        arg = {tupl.split('=')[0]: tupl.split('=')[1] for tupl in sys.argv[1:]}
        dt = DT.cs()
        nb = NB.cs()
        eq = Equalization.cs()
        pip2 = Pipeline.cs(config_spaces=[eq])
        pip1 = Pipeline.cs(config_spaces=[dt])
        sw = Switch.cs(config_spaces=[dt, nb])
        # pip1 = Pipeline.tree(config_spaces=[dt.tree()])
        # pip2 = Pipeline.tree(config_spaces=[pip1])
        print('configspace-----\n', pip1)
        # print('config dt =======\n', dt.tree().sample())
        print('config=======\n', pip1.sample())
        # pip3 = Pipeline(components=[])
        my_modelers = [dt]
        my_preprocessors = [pip2]

        for k, v in arg.items():
            print(f'{k}={v}')

        if 'storage' in arg:
            if arg['storage'] == 'sqlite':
                from paje.storage.sqlite import SQLite
                storage = SQLite(debug=not True)
            elif arg['storage'] == 'mysql':
                from paje.storage.mysql import MySQL
                storage = MySQL(db=arg['db'], debug=not True)
            elif arg['storage'] == 'dump':
                from paje.storage.pickledfile import PickledFile
                storage = PickledFile()
            elif arg['storage'] == 'cached':
                from paje.storage.mysql import MySQL
                from paje.storage.sqlite import SQLite
                storage = MySQL(db=arg['db'], nested=SQLite())
            elif arg['storage'] == 'sync':
                from paje.storage.mysql import MySQL
                from paje.storage.sqlite import SQLite
                storage = MySQL(db=arg['db'], nested=SQLite(),
                                sync=True)
            else:
                raise Exception('Wrong storage', arg['storage'])
        else:
            storage = None

        iterations = int(arg['iter']) if 'iter' in arg else 3
        random_state = int(arg['seed']) if 'seed' in arg else 0
        data = Data.read_arff(arg['data'], "class")

        trainset, testset = data.split(random_state=random_state)

        automl_rs = RandomAutoML(
            # preprocessors=default_preprocessors,
            # modelers=default_modelers,
            preprocessors=my_preprocessors,
            modelers=my_modelers,
            max_iter=iterations,
            pipe_length=2, repetitions=1,
            random_state=random_state,
            config={}
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
