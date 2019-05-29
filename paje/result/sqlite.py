import sqlite3

from paje.result.sql import SQL


class SQLite(SQL):
    def __init__(self, database='/tmp/paje.db', debug=False):
        self.database = database
        self.debug = debug

    def start(self):
        self.connection = sqlite3.connect(self.database)
        self.cursor = self.connection.cursor()

    def now_function(self):
        return 'datetime()'
