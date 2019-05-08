import sqlite3

from paje.result.storage import Cache


class SQLite(Cache):

    def __init__(self, database='/tmp/paje.db'):
        self.database = database

    def get_set(self, data):
        """
        Extract data from database.
        :param data:
        :return:
        """
        self.query("select data from dset where hash=?", [data.__hash__()])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                print('get_set: saindo do sqlite...')
                exit(0)
                # raise Exception('Excess of rows!')  # TODO: use general error handling
            return SQLite.unpack(rows[0][0])

    def setexists(self, data):
        """
        Check if data already exists in database.
        :param data:
        :return:
        """
        self.query("select count(1) from dset where hash=?", [data.__hash__()])
        rows = self.cursor.fetchall()
        return rows is not None and len(rows) > 0

    def argsexist(self, component):
        """
        Check if component args already exist in database.
        :param component:
        :return:
        """
        self.query("select count(1) from args where hash=?", [component.__hash__()])
        rows = self.cursor.fetchall()
        return rows is not None and len(rows) > 0

    def get_model(self, component, train):
        """
        Extract model from database.
        :param component:
        :param train:
        :return:
        """
        self.query("select name, args, dump from model where name=? and args=? and train=?",
                   [type(component.__class__()).__name__, component.__hash__(), train.__hash__()],
                   debug=False)
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            # for r in rows:
            #     print(r)
            # print()
            # [print(' row=', x[0]) for x in rows]
            # [print(' row type=', type(x[0])) for x in rows]
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                print('get_model: saindo do sqlite...')
                exit(0)
                # raise Exception('Excess of rows!')  # TODO: use general error handling
            # print('rows[0][2]: ', rows[0][2])
            # print('SQLite.unpack(rows[0][2]): ', SQLite.unpack(rows[0][2]))
            return SQLite.unpack(rows[0][2])

    def query(self, sql, args, debug=False):
        if debug:
            print(sql, args)
        try:
            self.cursor.execute(sql, args)
        except Exception as e:
            print(e)
            print()
            print(sql, args)
            exit(0)


    def get_or_else(self, component, train, f, test=None):
        # TODO: Repeated calls to this function with the same parameters can be memoized.
        # TODO: use test set for something

        # TODO: check if it is possible to put this creation part inside __init__ without locking database
        # TODO insert time spent
        self.connection = sqlite3.connect(self.database)
        self.cursor = self.connection.cursor()
        self.cursor.execute("create table if not exists args (hash BLOB, dic BLOB)")
        self.cursor.execute("create table if not exists dset (hash BLOB, data BLOB)")
        self.cursor.execute("create table if not exists model (name STRING, args BLOB, train BLOB, dump BLOB)")
        # self.cursor.execute("create table if not exists result (pipeline BLOB, train BLOB, test BLOB, prediction JSON)")
        # TODO: check if the size of the database is too big with dumps
        # TODO: replicability/cache of results: save testset and predictions
        self.cursor.execute("create table if not exists out (name STRING, args BLOB, setin BLOB, setout BLOB)")

        model = self.get_model(component, train)
        if model is not None:
            # Restoring from database.
            print('# Restoring from database.')
            component.model = model
            res = component.use(train)
        else:
            # Processing and inserting a new combination.
            train_hash = train.__hash__() # These two lines must be done, because train can be mutable during f().
            train_dump = SQLite.pack(train)
            if not self.setexists(train):
                self.query("insert into dset values (?, ?)",
                           [train_hash, train_dump])
            trainout = f(train)
            if not self.argsexist(component):
                self.query("insert into args values (?, ?)",
                           [component.__hash__(), component.serialized()])

            # Store model.
            dump = SQLite.pack(component.model)
            self.query("insert into model values (?, ?, ?, ?)",
                       [type(component.__class__()).__name__, component.__hash__(),
                        train_hash, dump], debug=False)

            # Store result of training data.
            if not self.setexists(trainout):
                self.query("insert into dset values (?, ?)",
                           [trainout.__hash__(), SQLite.pack(trainout)])
            # if test is not None:
            #     test_dump = SQLite.pack(test)
            #     self.cursor.execute("insert into set values (?, ?)",
            #                         [test.__hash__(), test_dump])
            self.query("insert into out values (?, ?, ?, ?)",
                       [type(component.__class__()).__name__,
                        component.__hash__(), train_hash, trainout.__hash__()])

            self.connection.commit()
            res = trainout

        self.connection.close()
        return res

    # def __del__(self):
    #     pass
