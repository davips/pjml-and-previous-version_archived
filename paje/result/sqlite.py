import sqlite3

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
        self.cursor.execute("create table if not exists result " +
                            "(idcomp BLOB, idtrain BLOB, idtest BLOB, " +
                            "trainout BLOB, testout BLOB, time FLOAT, model "
                            "BLOB)")
        # The idtest field is not strictly needed for now, may have some use.
        self.cursor.execute("CREATE INDEX if not exists idx_res ON result " +
                            "(idcomp, idtrain, idtest)")
        self.cursor.execute("create table if not exists args " +
                            "(idcomp BLOB, dic BLOB)")
        self.cursor.execute("CREATE INDEX if not exists idx_comp ON args " +
                            "(idcomp)")
        self.cursor.execute("CREATE INDEX if not exists idx_args ON args " +
                            "(dic)")
        # self.cursor.execute("create table if not exists model " +
        #                     "(name STRING, args BLOB, train BLOB, dump BLOB)")

    def get_result(self, component, train, test):
        """
        Look for a result in database.
        :param component:
        :param train:
        :param test:
        :return:
        """
        self.query(
            "select trainout, testout, model from result where "
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
            return unpack(rows[0][0]), unpack(rows[0][1])

    def store(self, component, train, test, trainout, testout, time):
        self.query("insert into result values (?, ?, ?, ?, ?, ?, ?)",
                   [component.uuid(), train.uuid, test.uuid,
                    pack(trainout), pack(testout), time, pack(component)])
        self.query("insert into args values (?, ?)",
                   [component.uuid(), component.serialized()])
        self.connection.commit()

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