import pymysql
import pymysql.cursors

from paje.result.sql import SQL


class MySQL(SQL):
    def __init__(self, database='root@143.107.183.114',
                 password='labotimo2019', db='curumim', debug=False):
        self.database = database
        self.password = password
        self.db = db
        self.user, self.host = database.split('@')
        self.debug = debug
        self.connection = pymysql.connect(host=self.host,
                                          user=self.user,
                                          password=self.password,
                                          charset='utf8mb4',
                                          cursorclass=pymysql.cursors.DictCursor)
        self.cursor = self.connection.cursor()
        self.cursor.execute("create database if not exists " + self.db)
        self.cursor.execute("use " + self.db)
        self.create_database()
