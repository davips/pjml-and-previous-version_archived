import socket
from abc import abstractmethod

from paje.base.data import Data
from paje.result.storage import Cache, unpack, pack


class SQL(Cache):
    def setup(self):
        if self.debug:
            print('creating tables...')
        self.query("create table if not exists args ("
                   f"id int NOT NULL primary key {self.auto_incr()}, "
                   "idcomp varchar(32) UNIQUE, "
                   "dic TEXT, "
                   "unique(dic(190)))")

        self.query("create table if not exists result ("
                   f"id int NOT NULL primary key {self.auto_incr()}, "
                   "idcomp varchar(32), idtrain varchar(32), idtest varchar(32)"
                   ", testout LONGBLOB"
                   ", timespent FLOAT"
                   ", dump LONGBLOB"
                   ", failed TINYINT"
                   ", start TIMESTAMP, end TIMESTAMP"
                   ", node varchar(32)"
                   ", UNIQUE(idcomp, idtrain, idtest))")
        self.query('CREATE INDEX idx1 ON result (timespent)')
        self.query('CREATE INDEX idx2 ON result (start)')
        self.query('CREATE INDEX idx3 ON result (end)')
        self.query('CREATE INDEX idx4 ON result (node)')

        self.query("create table if not exists dset ("
                   f"id int NOT NULL primary key {self.auto_incr()}, "
                   "iddset varchar(32) UNIQUE, "
                   "name varchar(256), "
                   "fields varchar(32), "
                   "data LONGBLOB, inserted timestamp)")
        self.query('CREATE INDEX idx5 ON dset (name(190))')
        self.query('CREATE INDEX idx6 ON dset (fields)')

        self.connection.commit()

    def lock(self, component, test):
        if self.debug:
            print('Locking...')
        node = socket.gethostname()
        txt = "insert into result values (null, ?, ?, ?, ?, ?, ?, ?, " + \
              self.now_function() + f", '0000-00-00 00:00:00', '{node}')"
        args = [component.uuid(), component.uuid_train, test.uuid(),
                None, None, None, 0]
        self.query(txt, args)
        self.connection.commit()

    def get_result(self, component, test):
        """
        Look for a result in database.
        :return:
        """
        self.query(
            "select testout, timespent, failed, end, node from result where "
            "idcomp=? and idtrain=? and idtest=?",
            [component.uuid(), component.uuid_train, test.uuid()])

        result = self._process_result()
        if result is None:
            return None

        dic = result[0] and unpack(result[0])
        testout = dic and \
                  test.sub(component.fields_to_keep_after_use()).updated(**dic)
        component.time_spent = result[1]
        component.failed = result[2] == 1
        component.locked = result[3] == '0000-00-00 00:00:00'
        component.node = result[4]
        return testout

    def store_data(self, data):
        if not self.data_exists(data):
            self.query("insert into dset values (NULL, ?, ?, ?, ?, "
                       f"{self.now_function()})",
                       [data.uuid(), data.name(),
                        data.fields_str(), data.dump])
            self.connection.commit()
        else:
            if self.debug:
                print('Testset already exists:' + data.uuid())

    def store(self, component, test, testout):
        """

        :param component:
        :param test:
        :param testout:
        :return:
        """
        slim = testout and testout.select(component.fields_to_store_after_use())
        # try:
        #     dump = pack(component)
        # except:
        # # except MemoryError as error:
        #     component.warning('Aborting dump storing due to memory issues.')
        #     from paje.module.modelling.classifier.nb import NB
        # TODO: dumps are not saved anymore!
        dump = None
        failed = 1 if component.failed else 0
        now = self.now_function()
        uuid_tr = component.uuid_train
        setters = f"testout=?, timespent=?, dump=?, failed=?"
        conditions = "idcomp=? and idtrain=? and idtest=?"
        self.query(f"update result set {setters}, start=start, end={now} "
                   f"where {conditions}",
                   [pack(slim), component.time_spent, dump, failed,
                    component.uuid(), uuid_tr, test.uuid()])

        if not self.component_exists(component):
            self.query("insert into args values (NULL, ?, ?)",
                       [component.uuid(), component.serialized()])
        else:
            component.warning(
                'Component already exists:' + str(component.serialized()))
        self.connection.commit()

        self.store_data(test)
        print('Stored!')

    def _process_result(self):
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            if len(rows) is not 1:
                for r in rows:
                    print(r)
                # TODO: use general error handling to show messages
                print('get_model: exiting sql...')
                exit(0)
            from paje.result.mysql import MySQL
            if isinstance(self, MySQL):
                return list(rows[0].values())
            else:
                return rows[0]

    def data_exists(self, data):
        return self.get_data(data, True) is not None

    def component_exists(self, component):
        return self.get_component(component, True) is not None

    def get_component(self, component, just_check_exists=False):
        field = 'dic'
        if just_check_exists:
            field = '1'
        self.query(f'select {field} from args where idcomp=?',
                   [component.uuid()])
        res = self._process_result()
        if res is None:
            return None
        else:
            return res[0]

    @staticmethod
    def interpolate(sql, lst0):
        lst = [str(w)[:100] for w in lst0]
        zipped = zip(sql.replace('?', '"?"').split('?'), map(str, lst + ['']))
        return ''.join(list(sum(zipped, ())))

    # @profile
    def query(self, sql, args=None):  # 300ms per query  (mozilla set)
        if args is None:
            args = []
        from paje.result.mysql import MySQL
        msg = self.interpolate(sql, args)
        if self.debug:
            print(msg)
        if isinstance(self, MySQL):
            sql = sql.replace('?', '%s')
            sql = sql.replace('insert or ignore', 'insert ignore')
            # self.connection.ping(reconnect=True)
        try:
            self.cursor.execute(sql, args)
        except Exception as e:
            print(e)
            print()
            print(msg)
            raise e

    def get_data(self, data, just_check_exists=False):
        return self.get_data_by_uuid(data.uuid(), just_check_exists)

    def get_data_by_uuid(self, datauuid, just_check_exists=False):
        field = '1' if just_check_exists else 'name, data'
        self.query(f'select {field} from dset where iddset=?', [datauuid])
        res = self._process_result()
        if res is None:
            return None
        else:
            return just_check_exists or Data(name=res[0], **unpack(res[1]))

    def get_component_dump(self, component):
        raise NotImplementedError('get model')

    @abstractmethod
    def now_function(self):
        pass

    @abstractmethod
    def auto_incr(self):
        pass

    def __del__(self):
        try:
            self.connection.close()
        except Exception as e:
            # print('Couldn\'t close database, but that\'s ok...', e)
            pass
