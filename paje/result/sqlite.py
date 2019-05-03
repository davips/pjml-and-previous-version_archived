import sqlite3

from paje.result.storage import Cache


class SQLite(Cache):

    def __init__(self, database='paje.db'):
        self.database = database
        self.connection = sqlite3.connect(database)
        self.cursor = self.connection.cursor()
        self.cursor.execute("create table if not exists args (hash BLOB, dic BLOB)")
        self.cursor.execute("create table if not exists dataset (hash BLOB, data BLOB)")
        # self.cursor.execute("create table if not exists result (pipeline BLOB, train BLOB, test BLOB, prediction JSON)")
        # TODO: check if the size of the database is too big with dumps
        # TODO: replicability/cache of results: save testset and predictions
        self.cursor.execute("create table if not exists model (args BLOB, train BLOB, dump BLOB, trainout BLOB)")  # args -> args.hash  train -> dataset.hash  trainout -> dataset.hash

    def get_or_else(self, component, train, f, test=None):
        # TODO: Repeated calls to this function with the same parameters can be memoized.
        self.cursor.execute("select dump, data from model, dataset where trainout=hash and args=? and train=?",
                            [component.__hash__(), train.__hash__()])
        rows = self.cursor.fetchall()
        for r in rows:
            print('row ', r)
        if rows is None or len(rows) == 0:
            # TODO: finish implementation. inserir datasets apenas se ainda nao existirem na base
            # Recover dump of data, if it already was stored in past summers.
            self.cursor.execute("select data from dataset where hash=?", [train.__hash__()])
            rows_data = self.cursor.fetchall()
            if rows_data is None or len(rows_data) == 0:
                trainout = f(train)
                trainout_dump = SQLite.pack(trainout)
            else:
                trainout_dump = rows_data[0]

            # Store model and result of training data.
            model_dump = SQLite.pack(component.model)
            self.cursor.execute("insert into model values (?, ?, ?, ?)",
                                [component.__hash__(), train.__hash__(), model_dump, trainout.__hash__()])
            self.cursor.execute("insert into args values (?, ?)",
                                [component.__hash__(), component.serialized()])
            self.cursor.execute("insert into dataset values (?, ?)",
                                [train.__hash__(), trainout_dump])
            if test is not None:
                test_dump = SQLite.pack(test)
                self.cursor.execute("insert into dataset values (?, ?)",
                                    [test.__hash__(), test_dump])
            self.connection.commit()
        else:
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                raise Exception('Excess of rows!')  # TODO: use general error handling
            print('saindo do sqlite...')
            [print(type(x)) for x in rows[0]]
            exit(0)
            component.model, trainout = [SQLite.unpack(x) for x in rows[0]]
        return trainout

    def __del__(self):
        self.connection.close()
