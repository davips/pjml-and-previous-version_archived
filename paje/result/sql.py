from abc import abstractmethod
from time import sleep

from paje.base.data import Data
from paje.result.storage import Cache, unpack, pack


class SQL(Cache):
    def setup(self):
        if self.debug:
            print('creating tables...')
        self.query("create table if not exists result ("
                   "idcomp varchar(32), idtrain varchar(32), idtest varchar(32)"
                   ", testout LONGBLOB, timespent FLOAT, dump LONGBLOB"
                   ", failed TINYINT"
                   ", start TIMESTAMP, end TIMESTAMP"
                   ", PRIMARY KEY(idcomp, idtrain, idtest))")
        # idtest field is not strictly needed for now, but may have some use.
        # self.cursor.execute("CREATE INDEX if not exists idx_res ON result "
        #                     "(idcomp, idtrain, idtest)")

        self.query("create table if not exists args "
                   "(idcomp varchar(32) PRIMARY KEY, dic TEXT)")
        # self.cursor.execute("CREATE INDEX if not exists idx_comp ON args "
        #                     "(idcomp)")
        # self.cursor.execute("CREATE INDEX if not exists idx_dic ON args "
        #                     "(dic)")

        self.query("create table if not exists dset "
                   "(iddset varchar(32) PRIMARY KEY, data LONGBLOB)")
        # self.cursor.execute("CREATE INDEX if not exists idx_dset ON dset "
        #                     "(iddset)")
        self.connection.commit()

    def lock(self, component, test):
        if self.debug:
            print('Locking...')
        txt = "insert into result values (?, ?, ?, ?, ?, ?, ?, " + \
              self.now_function() + ", '0000-00-00 00:00:00')"
        args = [component.uuid(), component.uuid_train, test.uuid,
                None, None, None, 0]
        self.query(txt, args)
        self.connection.commit()

    def get_result(self, component, test):
        """
        Look for a result in database.
        :return:
        """
        self.query(
            "select testout, timespent, failed, end from result where "
            "idcomp=? and idtrain=? and idtest=?",
            [component.uuid(), component.uuid_train, test.uuid])

        r = self._process_result()
        if r is None:
            return None
        else:
            data = r[0] and unpack(r[0])
            testout = data and test.sub(component.fields_to_keep_after_use()) \
                .updated(**data.fields)
            component.time_spent = r[1]
            component.failed = True if r[2] == 1 else False
            component.locked = True if r[3] == '0000-00-00 00:00:00' else False
            return testout

    def store_dset(self, data):
        print('1111111111', self.data_exists(data))
        if not self.data_exists(data):
            self.query("insert into dset values (?, ?)",
                       [data.uuid, pack(data)])
            self.connection.commit()
        else:
            if self.debug:
                print('Testset already exists:' + data.uuid)

    def store(self, component, test, testout):
        """

        :param component:
        :param test:
        :param testout:
        :return:
        """
        slimout = testout and testout.sub(component.fields_to_store_after_use())
        # try:
        #     dump = pack(component)
        # except:
        # # except MemoryError as error:
        #     component.warning('Aborting dump storing due to memory issues.')
        #     from paje.module.modelling.classifier.nb import NB
        # TODO: dumps are not saved anymore!
        dump = None
        fail = 1 if component.failed else 0
        now = self.now_function()
        uuid_tr = component.uuid_train
        setters = f"testout=?, timespent=?, dump=?, failed=?"
        conditions = "idcomp=? and idtrain=? and idtest=?"
        self.query(f"update result set {setters}, start=start, end={now} "
                   f"where {conditions}",
                   [pack(slimout), component.time_spent, dump, fail,
                    component.uuid(), uuid_tr, test.uuid])

        if not self.component_exists(component):
            self.query("insert into args values (?, ?)",
                       [component.uuid(), component.serialized()])
        else:
            component.warning(
                'Component already exists:' + str(component.serialized()))
        self.connection.commit()

        self.store_dset(test)
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
        return data is None or self.get_data(data, True) is not None

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
    def interpolate(sql, lst):
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
        field = 'data'
        if just_check_exists:
            field = '1'
        self.query(f'select {field} from dset where iddset=?', [data.uuid])
        res = self._process_result()
        if res is None:
            return None, None
        else:
            return Data(**unpack(res[0]))

    def get_component_dump(self, component):
        raise NotImplementedError('get model')

    @abstractmethod
    def now_function(self):
        pass

    def __del__(self):
        self.connection.close()

