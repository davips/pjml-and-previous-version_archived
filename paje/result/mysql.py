import pymysql
import pymysql.cursors

from paje.result.sql import SQL


class MySQL(SQL):
    def __init__(self, database='paje@143.107.183.114',
                 password='pajelanca19', db='curumim', debug=False):
        self.database = database
        self.password = password
        self.db = db
        self.user, self.host = database.split('@')
        self.debug = debug
        if self.debug:
            print('getting connection...')
        self.connection = pymysql.connect(host=self.host,
                                          user=self.user,
                                          password=self.password,
                                          charset='utf8mb4',
                                          cursorclass=pymysql.cursors.DictCursor)
        if self.debug:
            print('getting cursor...')
        self.cursor = self.connection.cursor()
        if self.debug:
            print('creating database', db, 'on', database, '...')
        self.cursor.execute("create database if not exists " + self.db)
        if self.debug:
            print('using database', db, 'on', database, '...')
        self.cursor.execute("use " + self.db)
