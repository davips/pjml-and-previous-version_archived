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
        self.cursor.execute("create table if not exists result " +
                            "(comp BLOB, train BLOB, setin BLOB, setout BLOB)")
        self.cursor.execute(
            "CREATE INDEX if not exists idx ON result (comp, train, setin)")
        # self.cursor.execute(
        #     "create table if not exists args (hash BLOB, dic BLOB)")
        # self.cursor.execute(
        #     "create table if not exists dset (hash BLOB, data BLOB)")
        # self.cursor.execute("create table if not exists model " +
        #                     "(name STRING, args BLOB, train BLOB, dump BLOB)")
        # self.cursor.execute(
        #     "CREATE INDEX if not exists idx2 ON dset (uuid)")

    def get_setout(self, component, train, setin):
        """
        Look for model in database.
        :param component:
        :param train:
        :param setin:
        :return:
        """
        self.query(
            "select setout from result where comp=? and train=? and setin=?",
            [component.uuid, train.uuid, setin.uuid])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                print('get_model: exiting sqlite...')
                exit(0)
                # TODO: use general error handling
            return setin.updated(z=unpack(rows[0][0]).z)

    def store(self, component, train, setin, setout):
        self.query("insert into result values (?, ?, ?, ?)",
                   [component.uuid, train.uuid, setin.uuid,
                    pack(Data(z=setout.z))])

        # # Store sets.
        # self.query("insert into dset values (?, ?)", [data.uuid,
        # pack(data.X)])

        # # Store model.
        # if not self.argsexist(component):
        #     self.query("insert into args values (?, ?)",
        #                [component.uuid, component.serialized()])
        #
        # dump = SQLite.pack(component.model)
        # self.query("insert into model values (?, ?, ?, ?)",
        #            [type(component.__class__()).__name__, component.uuid,
        #             train.uuid, dump])

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

    # def get_set(self, data_uuid):
    #     """
    #     Extract data from database.
    #     :param data_uuid:
    #     :return:
    #     """
    #     self.query("select data from dset where uuid=?", [data_uuid])
    #     rows = self.cursor.fetchall()
    #     if rows is None or len(rows) == 0:
    #         return None
    #     else:
    #         if len(rows) is not 1:
    #             for r in rows:
    #                 print(r)
    #             print('get_set: saindo do sqlite...')
    #             exit(0)
    #             # TODO: use general error handling
    #             # raise Exception('Excess of rows!')
    #         return unpack(rows[0][0])
    #
    # def setexists(self, data):
    #     """
    #     Check if data already exists in database.
    #     :param data:
    #     :return:
    #     """
    #     self.query("select 1 from dset where uuid=?", [data.uuid])
    #     rows = self.cursor.fetchall()
    #     if self.debug and rows is not None:
    #         print('Rows:')
    #         for row in rows:
    #             print(row)
    #     return rows is not None and len(rows) > 0
    #
    # def argsexist(self, component):
    #     """
    #     Check if component args already exist in database.
    #     :param component:
    #     :return:
    #     """
    #     self.query("select count(1) from args where hash=?",
    #                [component.uuid])
    #     rows = self.cursor.fetchall()
    #     return rows is not None and len(rows) > 0
    #
    #
    # def get_model(self, component, train):
    #     """
    #     Extract model from database.
    #     :param component:
    #     :param train:
    #     :return:
    #     """
    #     self.query(
    #         "select name, args, dump from model where " +
    #         "name=? and args=? and train=?",
    #         [type(component.__class__()).__name__, component.uuid, train.uuid])
    #     rows = self.cursor.fetchall()
    #     if rows is None or len(rows) == 0:
    #         return None
    #     else:
    #         # for r in rows:
    #         #     print(r)
    #         # print()
    #         # [print(' row=', x[0]) for x in rows]
    #         # [print(' row type=', type(x[0])) for x in rows]
    #         if len(rows) is not 1:
    #             for r in rows:
    #                 print(r)
    #             print('get_model: saindo do sqlite...')
    #             exit(0)
    #             # TODO: use general error handling
    #             # raise Exception('Excess of rows!')
    #         # print('rows[0][2]: ', rows[0][2])
    #         # print('SQLite.unpack(rows[0][2]): ', SQLite.unpack(rows[0][2]))
    #         return unpack(rows[0][2])

# # Store model.
# if not self.argsexist(component):
#     self.query("insert into args values (?, ?)",
#                [component.uuid, component.serialized()])
#
# dump = SQLite.pack(component.model)
# self.query("insert into model values (?, ?, ?, ?)",
#            [type(component.__class__()).__name__, component.uuid,
#             train.uuid, dump])
