import socket
from abc import abstractmethod

from paje.base.data import Data
from paje.result.storage import Cache, unpack_data


class SQL(Cache):
    def setup(self):
        if self.debug:
            print('creating tables...')
        self.query("create table if not exists args ("
                   f"id integer NOT NULL primary key {self.auto_incr()}, "
                   "idcomp varchar(32) NOT NULL UNIQUE, "
                   "dic TEXT NOT NULL)")
        self.query(f'CREATE INDEX idx0 ON args (dic{self.keylimit()})')

        self.query("create table if not exists result ("
                   f"id integer NOT NULL primary key {self.auto_incr()}, "
                   "idcomp varchar(32) NOT NULL, idtrain varchar(32) NOT NULL ,"
                   "idtest varchar(32) NOT NULL"
                   ", idtestout varchar(32), timespent FLOAT, dump LONGBLOB"
                   ", failed TINYINT NOT NULL, start TIMESTAMP NOT NULL"
                   ", end TIMESTAMP NOT NULL, alive TIMESTAMP NOT NULL"
                   ", node varchar(32) NOT NULL, attempts int NOT NULL"
                   ", UNIQUE(idcomp, idtrain, idtest))")
        self.query('CREATE INDEX idx1 ON result (timespent)')
        self.query('CREATE INDEX idx2 ON result (start)')
        self.query('CREATE INDEX idx3 ON result (end)')
        self.query('CREATE INDEX idx4 ON result (alive)')
        self.query('CREATE INDEX idx5 ON result (node)')
        self.query('CREATE INDEX idx6 ON result (attempts)')
        self.query("create table if not exists dset ("
                   f"id integer NOT NULL primary key {self.auto_incr()}, "
                   "iddset varchar(32) NOT NULL UNIQUE, "
                   "name varchar(158) NOT NULL, fields varchar(32) NOT NULL, "
                   "data LONGBLOB NOT NULL, inserted timestamp NOT NULL)")
        self.query(f'CREATE INDEX idx7 ON dset (name, fields)')
        self.query('CREATE INDEX idx8 ON dset (fields)')
        self.query('CREATE INDEX idx9 ON dset (inserted)')
        self.commit()

    def lock(self, component, test):
        if self.debug:
            print('Locking...')
        node = self.hostname
        txt = "insert into result values (null, " \
              "?, ?, ?, " \
              "?, ?, ?, " \
              "?, " + self.now_function() + f", '0000-00-00 00:00:00', " \
                  f"'0000-00-00 00:00:00', '{node}', 0)"
        args = [component.uuid(), component.uuid_train(), test.uuid(),
                None, None, None,
                0]
        self.query(txt, args)

    def get_result(self, component, test):
        """
        Look for a result in database.
        :return:
        """
        if component.failed or component.locked:
            return None
        self.query(
            "select data, timespent, failed, end, node, name "
            "from result left join dset on idtestout=iddset "
            "where idcomp=? and idtrain=? and idtest=?",
            [component.uuid(), component.uuid_train(), test.uuid()])

        result = self._process_result()
        if result is None:
            return None
        if result['data'] is not None:
            slim = Data(name=result['name'], **unpack_data(result['data']))
            testout = slim.merged(
                test.reduced_to(component.fields_to_keep_after_use()))
        else:
            testout = None
        component.time_spent = result['timespent']
        component.failed = result['failed'] == 1
        component.locked = result['end'] == '0000-00-00 00:00:00'
        component.node = result['node']
        return testout

    def store_data(self, data):
        if not self.data_exists(data):
            self.query("insert into dset values (NULL, "
                       "?, "
                       "?, ?, "
                       f"?, {self.now_function()})",
                       [data.uuid(),
                        data.name(), data.fields_str(),
                        data.dump()])
        else:
            if self.debug:
                print('Testset already exists:' + data.uuid(), data.name())

    def store(self, component, test, testout, train=None):
        """

        :param component:
        :param test:
        :param testout:
        :param train:
        :return:
        """
        slim = testout and \
               testout.reduced_to(component.fields_to_store_after_use())
        # TODO: try to store dumps again?
        dump = None
        failed = 1 if component.failed else 0
        now = self.now_function()
        uuid_tr = component.uuid_train()

        self.start_transaction()
        setters = f"idtestout=?, timespent=?, dump=?, failed=?"
        conditions = "idcomp=? and idtrain=? and idtest=?"
        self.query(f"update result set {setters}, start=start, "
                   f"end={now}, alive={now} "
                   f"where {conditions}",
                   [slim and slim.uuid(),
                    component.time_spent, dump, failed,
                    component.uuid(), uuid_tr, test.uuid()])
        if not self.component_exists(component):
            self.query("insert into args values (NULL, ?, ?)",
                       [component.uuid(), component.serialized()])
        else:
            component.msg(
                'Component already exists:' + str(component.serialized()))

        if train is not None:
            self.store_data(train)
        self.store_data(test)
        slim and self.store_data(slim)
        self.commit()
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
                raise Exception('More than 1 row found!')
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
        if just_check_exists:
            return True
        return res[field]

    @staticmethod
    def interpolate(sql, lst0):
        lst = [str(w)[:100] for w in lst0]
        zipped = zip(sql.replace('?', '"?"').split('?'), map(str, lst + ['']))
        return ''.join(list(sum(zipped, ())))

    def commit(self):
        if self.debug:
            print('commit')
        self.connection.commit()
        self.intransaction = False

    # @profile
    def query(self, sql, args=None):
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
            if not self.intransaction:
                self.connection.commit()
        except Exception as e:
            print(e)
            print()
            print(e, msg)
            raise Exception('\nMaybe you should call storage.open()')

    def get_data(self, data, just_check_exists=False):
        return self.get_data_by_uuid(data.uuid(), just_check_exists)

    def get_data_by_uuid(self, datauuid, just_check_exists=False):
        field = '1' if just_check_exists else 'name, data'
        self.query(f'select {field} from dset where iddset=?', [datauuid])
        res = self._process_result()
        if res is None:
            return None
        if just_check_exists:
            return True
        return just_check_exists or Data(name=res['name'],
                                         **unpack_data(res['data']))

    def get_data_after_apply_by_name(self, name, just_check_exists=False):
        field = '1' if just_check_exists else 'data'
        self.query(f'select {field} from dset where name=? order by id', [name])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        if just_check_exists:
            return True
        datats = Data(name=name, **unpack_data(rows[1]['data'])).shrink()
        data = Data(name=name, **unpack_data(rows[0]['data'])).merged(datats)
        return just_check_exists or data

    def get_data_by_name(self, name, fields=None,
                         just_check_exists=False):
        """
        To just recover an original data set you can pass fields='X,y'
        (case insensitive).
        'None' means to recover as many fields as stored at the moment.
        :param name:
        :param fields: if None, get completa Data including predictions if any
        :param just_check_exists:
        :return:
        """
        one = '1' if just_check_exists else 'data'
        restrict = '' if fields is None else f"and fields='{fields.upper()}'"
        self.query(f'select {one} from dset '
                   f'where name=? {restrict} order by id', [name])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        if just_check_exists:
            return True
        dic = {}
        for row in rows:
            dic.update(unpack_data(row['data']))
        return just_check_exists or Data(name=name, **dic)

    def get_finished(self):
        self.query('select name '
                   'from result join dset on idtrain=iddset '
                   "where end!='0000-00-00 00:00:00' and failed=0")
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            return [row['name'] for row in rows]

    def start_transaction(self):
        self.intransaction = True

    @abstractmethod
    def keylimit(self):
        pass

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
