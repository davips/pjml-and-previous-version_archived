import sys
from sys import argv

from paje.automl.random import RandomAutoML
from paje.base.data import Data
from paje.evaluator.metrics import Metrics
from paje.module.modules import default_modelers, default_preprocessors
# @profile
from paje.module.preprocessing.unsupervised.feature.transformer.drpca import \
    DRPCA
from paje.result.mysql import MySQL
from paje.result.sqlite import SQLite
from paje.module.modelling.classifier.svmc import SVMC


def main():
    if len(sys.argv[1:]) < 1 or any(['=' not in k for k in sys.argv[1:]]):
        print('Usage: \npython toy.py data=/tmp/dataset.arff '
              '[iter=#] [seed=#] [storage=mysql/sqlite] [db=teste] ')
    else:
        arg = {tupl.split('=')[0]: tupl.split('=')[1] for tupl in sys.argv[1:]}
        for k, v in arg.items():
            print(f'{k}={v}')

        if 'storage' in arg:
            if arg['storage'] == 'sqlite':
                storage = SQLite(debug=not True)
            if arg['storage'] == 'mysql':
                storage = MySQL(db=arg['db'], debug=not True)
        else:
            storage = None

        iterations = int(arg['iter']) if 'iter' in arg else 3
        random_state = int(arg['seed']) if 'seed' in arg else 0
        data = Data.read_arff(arg['data'], "class")

        trainset, testset = data.split()

        automl_rs = RandomAutoML(
            preprocessors=default_preprocessors,
            modelers=default_modelers,
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

    # automl_rs = RandomAutoML(max_iter=n, static=False, fixed=True,
    #                          max_depth=4, repetitions=0, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()
    #
    # automl_rs = RandomAutoML(max_iter=n, static=False, fixed=True,
    #                          max_depth=10, repetitions=2, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()
    #
    # automl_rs = RandomAutoML(max_iter=n, static=False, fixed=False,
    #                          max_depth=8, repetitions=0, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()
    #
    # automl_rs = RandomAutoML(max_iter=2, static=False, fixed=False,
    #                          max_depth=10, repetitions=2, method="all")
    # automl_rs.apply(data_train)
    # resp = automl_rs.use(data_test).data_y
    # print("Accuracy score", sklearn.metrics.accuracy_score(data_test.data_y, automl_rs.use(data_test).data_y))
    # print()

    # print(1 - np.sum(resp == data_test.data_y)/resp.shape[0])
