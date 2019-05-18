import sqlite3

from paje.base.exceptions import ExceptionInApplyOrUse
from paje.result.storage import Cache


class SQLite(Cache):

    def __init__(self, database='/tmp/paje.db', debug=False):
        self.database = database
        self.debug = debug
        self.start_database()

    # @profile
    def get_set(self, data_hash):
        """
        Extract data from database.
        :param data:
        :return:
        """
        self.query("select data from dset where hash=?", [data_hash])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                print('get_set: saindo do sqlite...')
                exit(0)
                # TODO: use general error handling
                # raise Exception('Excess of rows!')
            return SQLite.unpack(rows[0][0])

    def setexists(self, data_hash):
        """
        Check if data already exists in database.
        :param data:
        :return:
        """
        self.query("select 1 from dset where hash=?", [data_hash],
                   debug=self.debug)
        rows = self.cursor.fetchall()
        if self.debug and rows is not None:
            print('Rows:')
            for row in rows:
                print(row)
        return rows is not None and len(rows) > 0

    def argsexist(self, component):
        """
        Check if component args already exist in database.
        :param component:
        :return:
        """
        self.query("select count(1) from args where hash=?",
                   [component.uuid])
        rows = self.cursor.fetchall()
        return rows is not None and len(rows) > 0

    # @profile
    def getsetout(self, component, train_hash, setin_hash):
        """
        Look for model in database.
        :param component:
        :param train:
        :return:
        """
        self.query(
            "select setout from out where " +
            "name=? and args=? and train=? and setin=?",
            [type(component.__class__()).__name__, component.uuid, train_hash,
             setin_hash], debug=self.debug)
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                print('get_model: saindo do sqlite...')
                exit(0)
                # TODO: use general error handling

            return self.get_set(rows[0][0])

    def get_model(self, component, train_hash):
        """
        Extract model from database.
        :param component:
        :param train:
        :return:
        """
        self.query(
            "select name, args, dump from model where name=? and args=? and train=?",
            [type(component.__class__()).__name__, component.uuid,
             train_hash],
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
                # TODO: use general error handling
                # raise Exception('Excess of rows!')
            # print('rows[0][2]: ', rows[0][2])
            # print('SQLite.unpack(rows[0][2]): ', SQLite.unpack(rows[0][2]))
            return SQLite.unpack(rows[0][2])

    def start_database(self):
        # TODO: check if it is possible to put this creation part inside
        #  __init__ without locking database
        self.connection = sqlite3.connect(self.database)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            "create table if not exists args (hash BLOB, dic BLOB)")
        self.cursor.execute(
            "create table if not exists dset (hash BLOB, data BLOB)")
        self.cursor.execute("create table if not exists model " +
                            "(name STRING, args BLOB, train BLOB, dump BLOB)")
        self.cursor.execute("create table if not exists out " +
                            "(name STRING, args BLOB, train BLOB, setin BLOB, "
                            "setout BLOB)")
        self.cursor.execute(
            "CREATE INDEX if not exists idx2 ON dset (hash)")
        self.cursor.execute(
            "CREATE INDEX if not exists idx ON out (name, args, train, setin)")

    def get_or_else(self, component, train, f):
        # TODO: Repeated calls to this function with the same parameters can be
        #  memoized.

        # TODO: check if it is possible to put this creation part inside
        #  __init__ without locking database
        # TODO insert time spent
        self.connection = sqlite3.connect(self.database)
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            "create table if not exists args (hash BLOB, dic BLOB)")
        self.cursor.execute(
            "create table if not exists dset (hash BLOB, data BLOB)")
        self.cursor.execute(
            "create table if not exists model (name STRING, args BLOB, train BLOB, dump BLOB)")
        # self.cursor.execute("create table if not exists result (pipeline BLOB, train BLOB, test BLOB, prediction JSON)")
        # TODO: check if the size of the database is too big with dumps
        # TODO: replicability/cache of results: save testset and predictions
        self.cursor.execute(
            "create table if not exists out (name STRING, args BLOB, setin BLOB, setout BLOB)")

        model = self.get_model(component, train)
        if model is not None:
            # Restoring from database.
            # print('# Restoring from database.')
            component.model = model
            res = component.use(train)
        else:
            print('memoizing...')

            # Processing and inserting a new combination.
            train_dump = SQLite.pack(train)
            if not self.setexists(train.uuid):
                self.query("insert into dset values (?, ?)",
                           [train.uuid, train_dump])

            # apply()
            try:
                trainout = f(train)
            except Exception as e:
                raise ExceptionInApplyOrUse(e)

            if not self.argsexist(component):
                self.query("insert into args values (?, ?)",
                           [component.uuid, component.serialized()])

            # Store model.
            dump = SQLite.pack(component.model)
            self.query("insert into model values (?, ?, ?, ?)",
                       [type(component.__class__()).__name__,
                        component.uuid,
                        train.uuid, dump], debug=False)

            # Store result of training data.
            if not self.setexists(trainout.uuid):
                self.query("insert into dset values (?, ?)",
                           [trainout.uuid, SQLite.pack(trainout)])
            # if test is not None:
            #     test_dump = SQLite.pack(test)
            #     self.cursor.execute("insert into set values (?, ?)",
            #                         [test.uuid, test_dump])
            self.query("insert into out values (?, ?, ?, ?)",
                       [type(component.__class__()).__name__,
                        component.uuid, train.uuid, trainout.uuid])

            self.connection.commit()
            res = trainout
        return res

    # @profile
    def get_results_or_else(self, component, train, setin, f):
        # TODO: Repeated calls to this function with the same parameters can
        #  be memoized, to avoid network delays, for instance.
        # TODO insert time spent
        setout = self.getsetout(component, train.uuid, setin.uuid)
        if setout is None:
            print('memoizing results...')
            # TODO: is it useful to store the dump of the sets?

            # Apply f()
            try:
                setout = f(setin)
                setout_hash = setout.uuid
            except Exception as e:
                print('function:', f)
                print('shape train:', train.X.shape, train.y.shape)
                print('shape setin:', setin.X.shape, setin.y.shape)
                raise ExceptionInApplyOrUse(e)

            # Store result.
            # TODO: create insertset()
            if not self.setexists(setout_hash):
                setoutdump = SQLite.pack(setout)
                self.query("insert into dset values (?, ?)",
                           [setout_hash, setoutdump], debug=self.debug)
            self.query("insert into out values (?, ?, ?, ?, ?)",
                       [type(component.__class__()).__name__,
                        component.uuid, train.uuid, setin.uuid,
                        setout_hash], debug=self.debug)

            self.connection.commit()
        return setout

    # def __del__(self):
    #     pass

    def query(self, sql, args, debug=False):
        if debug:
            print(sql, args)
        try:
            self.cursor.execute(sql, args)
        except Exception as e:
            print(e)
            print()
            print(sql, args)
            raise e

    def __del__(self):
        self.connection.close()
