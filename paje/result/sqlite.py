import sqlite3

from paje.base.data import Data
from paje.result.storage import Cache, unpack, pack


class SQLite(Cache):

    def __init__(self, database='/tmp/paje.db', debug=False):
        self.database = database
        self.debug = debug
        self.start_database()

    def start_database(self):
        # TODO: check if it is possible to put this creation part inside
        #  __init__ without locking database
        self.connection = sqlite3.connect(self.database)
        self.cursor = self.connection.cursor()

        self.cursor.execute("create table if not exists result "
                            "(idcomp BLOB, idtrain BLOB, idtest BLOB, "
                            "trainout BLOB, testout BLOB, time FLOAT, "
                            "model BLOB, PRIMARY KEY(idcomp, idtrain, idtest))")
        # idtest field is not strictly needed for now, but may have some use.
        # self.cursor.execute("CREATE INDEX if not exists idx_res ON result "
        #                     "(idcomp, idtrain, idtest)")

        self.cursor.execute("create table if not exists args "
                            "(idcomp BLOB PRIMARY KEY, dic BLOB UNIQUE)")
        # self.cursor.execute("CREATE INDEX if not exists idx_comp ON args "
        #                     "(idcomp)")
        # self.cursor.execute("CREATE INDEX if not exists idx_dic ON args "
        #                     "(dic)")

        self.cursor.execute("create table if not exists dset "
                            "(iddset BLOB PRIMARY KEY, data BLOB UNIQUE)")
        # self.cursor.execute("CREATE INDEX if not exists idx_dset ON dset "
        #                     "(iddset)")

    def get_result(self, component, train, test):
        """
        Look for a result in database.
        :param component:
        :param train:
        :param test:
        :return:
        """
        self.query(
            "select trainout, testout from result where "
            "idcomp=? and idtrain=? and idtest=?",
            [component.uuid(), train.uuid, test.uuid])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None, None
        else:
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                # TODO: use general error handling to show messages
                print('get_model: exiting sqlite...')
                exit(0)
            return train.updated(**unpack(rows[0][0]).predictions), \
                   test.updated(**unpack(rows[0][1]).predictions)

    def store(self, component, train, test, trainout, testout, time_spent):
        slim_trainout = Data(**trainout.predictions)
        slim_testout = Data(**testout.predictions)
        self.query("insert into result values (?, ?, ?, ?, ?, ?, ?)",
                   [component.uuid(), train.uuid, test.uuid,
                    pack(slim_trainout), pack(slim_testout), time_spent,
                    pack(component)])
        self.query("insert or ignore into args values (?, ?)",
                   [component.uuid(), component.serialized()])
        self.query("insert or ignore into dset values (?, ?)", [train.uuid,
                                                                pack(train)])
        self.query("insert or ignore into dset values (?, ?)",
                   [test.uuid, pack(test)])
        self.connection.commit()

    def get_model(self, component, train, test):
        raise NotImplementedError('get model')

    def query(self, sql, args):
        if self.debug:
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
