from abc import abstractmethod
from sqlite3 import IntegrityError as IntegrityErrorSQLite

from paje.base.data import Data
from paje.result.storage import Cache, unpack_data
from pymysql import IntegrityError as IntegrityErrorMySQL


class SQL(Cache):
    def setup(self):
        if self.debug:
            print('creating tables...')

        self.query(f'''
            create table if not exists name (
                n integer NOT NULL primary key {self.auto_incr()},

                nid char(32) NOT NULL UNIQUE,
                txt TEXT NOT NULL
            )''')
        self.query(f'CREATE INDEX nam0 ON name (txt{self.keylimit()})')

        self.query(f'''
            create table if not exists dump (
                n integer NOT NULL primary key {self.auto_incr()},

                duid char(32) NOT NULL UNIQUE,
                bytes LONGBLOB NOT NULL
            )''')

        self.query(f'''
            create table if not exists log (
                n integer NOT NULL primary key {self.auto_incr()},

                lid char(32) NOT NULL UNIQUE,
                msg TEXT NOT NULL,
                insl timestamp NOT NULL
            )''')
        self.query(f'CREATE INDEX log0 ON log (msg{self.keylimit()})')
        self.query(f'CREATE INDEX log1 ON log (insl{self.keylimit()})')

        self.query(f'''
            create table if not exists com (
                n integer NOT NULL primary key {self.auto_incr()},
                
                cid char(32) NOT NULL UNIQUE,
                arg TEXT NOT NULL,
                
                insc timestamp NOT NULL
            )''')
        self.query(f'CREATE INDEX com0 ON com (arg{self.keylimit()})')
        self.query(f'CREATE INDEX com1 ON com (insc)')

        self.query(f'''
            create table if not exists data (
                n integer NOT NULL primary key {self.auto_incr()},

                fields varchar(32) NOT NULL,
                name char(32) NOT NULL,

                did char(32) NOT NULL,  <---- uuid(name + fields)
                shape varchar(32) NOT NULL,

                insd timestamp NOT NULL,

                UNIQUE(fields, name),
                FOREIGN KEY (did) REFERENCES dump(duid),
               )''')
        self.query(f'CREATE INDEX data0 ON data (shape{self.keylimit()})')
        self.query(f'CREATE INDEX data1 ON data (insd)')

        self.query(f'''
            create table if not exists res (
                n integer NOT NULL primary key {self.auto_incr()},

                node varchar(32) NOT NULL,

                com char(32) NOT NULL,
                dtr char(32) NOT NULL,
                din char(32) NOT NULL,
                log char(32),

                dout char(32),
                spent FLOAT,
                dumpr char(32),

                fail TINYINT,

                start TIMESTAMP NOT NULL,
                end TIMESTAMP NOT NULL,
                alive TIMESTAMP NOT NULL,

                tries int NOT NULL,
                locks int NOT NULL,

                UNIQUE(com, dtr, din),
                FOREIGN KEY (com) REFERENCES com(cid),
                FOREIGN KEY (dtr) REFERENCES data(did),
                FOREIGN KEY (din) REFERENCES data(did),
                FOREIGN KEY (dumpr) REFERENCES dump(duid),
                FOREIGN KEY (log) REFERENCES log(lid)
            )''')
        self.query('CREATE INDEX res0 ON res (dout)')
        self.query('CREATE INDEX res1 ON res (spent)')
        self.query('CREATE INDEX res2 ON res (dumpr)')
        self.query('CREATE INDEX res3 ON res (fail)')
        self.query('CREATE INDEX res4 ON res (start)')
        self.query('CREATE INDEX res5 ON res (end)')
        self.query('CREATE INDEX res6 ON res (alive)')
        self.query('CREATE INDEX res7 ON res (node)')
        self.query('CREATE INDEX res8 ON res (tries)')
        self.query('CREATE INDEX res9 ON res (locks)')
        self.query('CREATE INDEX res10 ON res (log)')

    def lock(self, component, input_data, txtres=''):
        """
        Store 'test' and 'component' if they are not yet stored.
        Insert a locking row corresponding to comp,training_data,input_data.
        :param component:
        :param input_data:
        :param txtres: for logging purposes
        :return:
        """
        if self.debug:
            print('Locking...')

        # Store testing set if (inexistent yet).
        self.store_data(
            input_data.reduced_to(component.fields_to_keep_after_use()))

        # Store component (if inexistent yet) and attempt to acquire lock.
        # Mark as locked_by_others otherwise.
        nf = self.now_function()
        args_comp = [component.uuid(), component.serialized()]
        args_res = [self.hostname,
                    component.uuid(), component.uuid_train(), input_data.uuid()]
        sql = f'''
            begin;
            insert or ignore into com values (
                NULL,
                ?, ?,
                {nf}
            );
            insert into res values (
                null,
                ?
                ?, ?, ?, null
                null, null, null,
                null,
                {nf}, '0000-00-00 00:00:00', '0000-00-00 00:00:00',
                0, 0
            end;'''
        try:
            self.query(sql, args_comp + args_res)
        except IntegrityErrorSQLite as e:
            print(f'Unexpected lock! Giving up my turn on {txtres}', e)
            component.locked_by_others = True
        except IntegrityErrorMySQL as e:
            print(f'Unexpected lock! Giving up my turn on {txtres}', e)
            component.locked_by_others = True
        else:
            component.locked_by_others = False
            print(f'Locked [{txtres}]!')

    def get_result(self, component, test):
        """
        Look for a result in database.
        :return:
        """
        if component.failed or component.locked_by_others:
            return None
        self.query(f'''
            select 
                bytes, spent, fail, end, node, name
            from 
                result left join dset on dout = did
                    left join dump on dumpd = duid
                    left join name on name = nid
            where                
                cid=? and dtr=? and dts=?''',
                   [component.uuid(), component.uuid_train(), test.uuid()])

        result = self._process_result()
        if result is None:
            return None
        if result['bytes'] is not None:
            slim = Data(name=result['name'], **unpack_data(result['bytes']))
            testout = slim.merged(
                test.reduced_to(component.fields_to_keep_after_use()))
        else:
            testout = None
        component.time_spent = result['spent']
        component.failed = result['fail'] and result['fail'] == 1
        component.locked_by_others = result['end'] == '0000-00-00 00:00:00'
        component.node = result['node']
        return testout

    def store_data(self, data):
        # Check first with a low cost query if data already exists.
        if self.data_exists(data):
            if self.debug:
                print('Data already exists:' + data.uuid(), data.name())
            return

        # Insert dump of data and data info.
        # Ignore insertion if the event of another job winning the race.
        sql = f'''
            begin;
            insert or ignore into dump values (
                null,
                ?, ?
            );
            insert or ignore into data values (
                NULL,
                ?, ?, 
                ?, ?,
                {self.now_function()}
            );
            end;'''
        dump_args = [data.uuid(), data.dump()]
        data_args = [data.uuid(), data.fields(), data.name(),
                     data.shapes()]
        try:
            self.query(sql, dump_args + data_args)
        except IntegrityErrorSQLite as e:
            print(f'Data already store before!', data.uuid())
        except IntegrityErrorMySQL as e:
            print(f'Data already store before!', data.uuid())
        else:
            print(f'Data inserted', data.uuid())

    def store(self, component, test, testout):
        """

        :param component:
        :param test:
        :param testout:
        :param train:
        :return:
        """
        slim = testout and \
               testout.reduced_to(component.fields_to_store_after_use())
        # TODO: try to store component dumps again?
        dump = None
        failed = 1 if component.failed else 0
        now = self.now_function()
        uuid_tr = component.uuid_train()

        # TODO: is there any important exception to handle here?
        sql = f'''
            update result set 
                idtestout=?, timespent=?, dump=?, failed=?,
                 start=start, "
                   f"end={now}, alive={now} "
                   f"where 
                   idcomp=? and idtrain=? and idtest=?'''
        args = [slim and slim.uuid(),
                component.time_spent, dump, failed,
                component.uuid(), uuid_tr, test.uuid()]

        slim and self.store_data(slim)
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

    # @profile
    def query(self, sql, args=None):
        if self.read_only and not sql.startswith('select '):
            print('========================================\n',
                  'Attempt to write onto read-only storage!', sql)
            self.cursor.execute('select 1')
            return
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

        import sys
        import traceback

        try:
            self.cursor.execute(sql, args)
            self.connection.commit()
        except Exception as ex:
            # From StackOverflow
            msg = self.info + msg
            # Gather the information from the original exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            # Format the original exception for a nice printout:
            traceback_string = ''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback))
            # Re-raise a new exception of the same class as the original one
            raise type(ex)(
                "%s\norig. trac.:\n%s\n" % (msg, traceback_string))

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
        one = '1' if just_check_exists else 'data,iddset'
        if fields is None:
            self.query(f'select {one} from dset '
                       f'where name=? order by id', [name])
        else:
            self.query(f'select {one} from dset where '
                       f'name=? and fields=? order by id', [name, fields])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        if just_check_exists:
            return True
        dic = {}
        for row in rows:
            dic.update(unpack_data(row['data']))
        if len(rows) > 1:
            data = Data(name=name, **dic)
        else:
            data = Data.with_uuid(uuid=rows[0]['iddset'], name=name, **dic)
        return just_check_exists or data

    def get_data_uuid_by_name(self, name, fields='X,y',
                              just_check_exists=False):
        """
        UUID for a combination of name and fields.
        :param name:
        :param fields: which view of Data we are being asked for.
        :param just_check_exists:
        :return:
        """
        one = '1' if just_check_exists else 'iddset'
        self.query(f"select {one} from dset where "
                   f"name=? and fields=? order by id", [name, fields.upper()])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        if just_check_exists:
            return True
        if len(rows) > 1:
            raise Exception(f'Excess of rows for {name} {fields}!',
                            f'{len(rows)} > 1', rows)
        return just_check_exists or rows[0]['iddset']

    def get_finished(self):  # TODO: specify search criteria (by component?)
        self.query('select name '
                   'from result join dset on idtrain=iddset '
                   "where end!='0000-00-00 00:00:00' and failed=0")
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            return [row['name'] for row in rows]

    def count_results(self, component, data):
        self.query('select id from result where idcomp=? and idtrain=?',
                   [component.uuid(), data.uuid()])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return 0
        else:
            return len(rows)

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
