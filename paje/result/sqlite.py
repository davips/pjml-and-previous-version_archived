import socket
import sqlite3

from paje.result.sql import SQL


class SQLite(SQL):
    def __init__(self, database='/tmp/paje.db', debug=False, read_only=False):
        self.info = database
        self.read_only = read_only
        self.hostname = socket.gethostname()
        self.database = database
        self.debug = debug
        self.intransaction = False
        self.open()

    def open(self):
        self.connection = sqlite3.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

        # Create tables if they don't exist yet.
        try:
            self.query(f"select 1 from result")
        except:
            if self.debug:
                print('creating database', self.database, '...')
            self.setup()

    def now_function(self):
        return 'datetime()'

    def auto_incr(self):
        return 'AUTOINCREMENT'

    def keylimit(self):
        return ''
