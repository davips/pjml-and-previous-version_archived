import socket
import sqlite3

from paje.result.sql import SQL


class SQLite(SQL):
    def __init__(self, database='/tmp/paje.db', debug=False, read_only=False,
                 nested_storage=None):
        super().__init__(nested_storage=nested_storage)
        self.info = database
        self.read_only = read_only
        self.hostname = socket.gethostname()
        self.database = database
        self.debug = debug
        self._open()

    def _open(self):
        # isolation_level=None -> SQLite autocommiting
        self.connection = sqlite3.connect(self.database,
                                          isolation_level="DEFERRED")
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()

        # Create tables if they don't exist yet.
        try:
            self.query(f"select 1 from res")
        except:
            if self.debug:
                print('creating database', self.database, '...')
            self._setup()

    def _now_function(self):
        return 'datetime()'

    def _auto_incr(self):
        return 'AUTOINCREMENT'

    def _keylimit(self):
        return ''

    def _on_conflict(self, fields=''):
        return f'ON CONFLICT{fields} DO UPDATE SET'