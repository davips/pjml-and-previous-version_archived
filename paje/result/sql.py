from abc import abstractmethod
from sqlite3 import IntegrityError as IntegrityErrorSQLite

from paje.base.data import Data
from paje.result.storage import Cache, unpack_data
from pymysql import IntegrityError as IntegrityErrorMySQL


class SQL(Cache):
    def setup(self):
        if self.debug:
            print('creating tables...')

        # History of Data
        # ========================================================
        self.query(f'''
            create table if not exists hist (
                n integer NOT NULL primary key {self.auto_incr()},

                hid char(19) NOT NULL UNIQUE,

                txt TEXT NOT NULL,
            )''')
        self.query(f'CREATE INDEX nam0 ON hist (txt{self.keylimit()})')

        # Names of Data ========================================================
        self.query(f'''
            create table if not exists name (
                n integer NOT NULL primary key {self.auto_incr()},

                nid char(19) NOT NULL UNIQUE,

                des TEXT NOT NULL,

                cols TEXT
            )''')
        self.query(f'CREATE INDEX nam0 ON name (des{self.keylimit()})')
        self.query(f'CREATE INDEX nam1 ON name (cols{self.keylimit()})')

        # Dumps of Data and Component ==========================================
        self.query(f'''
            create table if not exists dump (
                n integer NOT NULL primary key {self.auto_incr()},

                duid char(19) NOT NULL UNIQUE,

                bytes LONGBLOB NOT NULL
            )''')

        # Logs for Component ===================================================
        self.query(f'''
            create table if not exists log (
                n integer NOT NULL primary key {self.auto_incr()},

                lid char(19) NOT NULL UNIQUE,

                msg TEXT NOT NULL,
                insl timestamp NOT NULL
            )''')
        self.query(f'CREATE INDEX log0 ON log (msg{self.keylimit()})')
        self.query(f'CREATE INDEX log1 ON log (insl{self.keylimit()})')

        # Components ===========================================================
        self.query(f'''
            create table if not exists com (
                n integer NOT NULL primary key {self.auto_incr()},
                
                cid char(19) NOT NULL UNIQUE,

                arg TEXT NOT NULL,
                
                insc timestamp NOT NULL
            )''')
        self.query(f'CREATE INDEX com0 ON com (arg{self.keylimit()})')
        self.query(f'CREATE INDEX com1 ON com (insc)')

        # Datasets =============================================================
        self.query(f'''
            create table if not exists data (
                n integer NOT NULL primary key {self.auto_incr()},

                did char(19) NOT NULL UNIQUE,

                name char(19) NOT NULL,
                fields varchar(32) NOT NULL,
                hist char(19),

                dumpd char(19) NOT NULL,
                shape varchar(256) NOT NULL,

                insd timestamp NOT NULL,

                unique(name, fields, hist),
                FOREIGN KEY (name) REFERENCES name(nid),
                FOREIGN KEY (dumpd) REFERENCES dump(duid),
                FOREIGN KEY (hist) REFERENCES hist(hid)
               )''')
        # apontar pro data anterior? <- nao precisa pq estah na 'res'
        # mostrar comp <-nao adianta pq o msm comp pode ser aplicado varias vezes
        # provav. history não vai conter comps inuteis como pipes e switches
        self.query(f'CREATE INDEX data0 ON data (shape{self.keylimit()})')
        self.query(f'CREATE INDEX data1 ON data (insd)')
        self.query(f'CREATE INDEX data2 ON data (dumpd)')
        self.query(f'CREATE INDEX data3 ON data (fields)')  # needed?
        self.query(f'CREATE INDEX data4 ON data (name)')  # needed?
        self.query(f'CREATE INDEX data5 ON data (hist)')  # needed?

        # Results ==============================================================
        self.query(f'''
            create table if not exists res (
                n integer NOT NULL primary key {self.auto_incr()},

                node varchar(19) NOT NULL,

                com char(19) NOT NULL,
                dtr char(19) NOT NULL,
                din char(19) NOT NULL,
                log char(19),

                dout char(19),
                spent FLOAT,
                dumpc char(19),

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
                FOREIGN KEY (dout) REFERENCES data(did),
                FOREIGN KEY (dumpc) REFERENCES dump(duid),
                FOREIGN KEY (log) REFERENCES log(lid)
            )''')
        self.query('CREATE INDEX res0 ON res (dout)')
        self.query('CREATE INDEX res1 ON res (spent)')
        self.query('CREATE INDEX res2 ON res (dumpc)')
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
                ?,
                ?,
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

    def get_one(self):
        row = self.cursor.fetchone()
        if row is None:
            return None
        row2 = self.cursor.fetchone()
        if row2 is not None:
            raise Exception('Excess of rows, after ', row, ':', row2)
        return row

    def get_result(self, component, test):
        """
        Look for a result in database.
        :return:
        """
        if component.failed or component.locked_by_others:
            return None
        self.query(f'''
            select 
                des, bytes, spent, fail, end, node
            from 
                result 
                    left join dset on dout = did
                    left join dump on dumpd = duid
                    left join name on name = nid
            where                
                cid=? and dtr=? and dts=?''',
                   [component.uuid(), component.uuid_train(), test.uuid()])

        result = self.get_one()
        if result['bytes'] is not None:
            slim = Data(name=result['des'], **unpack_data(result['bytes']))
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
        # Catch exception in the event of another job winning the race.
        # Dumps can have duplicate, e.g. when two models give the same
        # predictions (usually in the same dataset), so 'insert or ignore'.
        # In the case of descriptions (e.g. only X,y) the same Data can be
        # inserted twice by different components, but this is checked above
        # by data_exists. If in the meantime, while the interpreter is
        # reading this comment, someone else insert just the same Data,
        # we will have to handle it as an exception.
        # ps. : in the presence of predictions, UUID will change accordingly,
        #  otherwise, it will depend only on 'Data.name' and 'Data.fields'.
        sql = f'''
            begin;
            insert or ignore into dump values (
                null,
                ?,
                ?
            );
            insert or ignore into name values (
                NULL,
                ?,
                ?,
                NULL
            );
            insert or ignore into hist values (
                NULL,
                ?,
                ?,
            );
            insert into data values (
                NULL,
                ?,
                ?, ?,
                ?, ?,
                {self.now_function()}
            );
            end;'''
        dump_args = [data.dump_uuid(),
                     data.dump()]
        name_args = [data.name_uuid(),
                     data.name()]
        hist_args = [data.hist_uuid(),
                     data.history()]
        data_args = [data.uuid(),
                     data.name(), data.fields(),
                     data.dump_uuid(), data.shapes()]
        try:
            self.query(sql, dump_args + name_args + hist_args + data_args)
        except IntegrityErrorSQLite as e:
            print(f'Data already store before!', data.uuid())
        except IntegrityErrorMySQL as e:
            print(f'Data already store before!', data.uuid())
        else:
            print(f'Data inserted', data.uuid())

    def store(self, component, input_data, testout):
        """

        :param component:
        :param input_data:
        :param testout:
        :param train:
        :return:
        """
        # Store resulting Data
        slim = testout and testout.reduced_to(
            component.fields_to_store_after_use())
        slim and self.store_data(slim)

        # Remove lock and point result to data inserted above.
        # TODO: try to store component dumps again?
        # TODO: is there any important exception to handle here?
        # We should set all timestamp fields even if with the same old value.
        now = self.now_function()
        sql = f'''
            update result set 
                dout=?, spent=?, dumpc=?,
                fail=?,
                start=start, end={now}, alive={now}
            where
                com=? and dtr=? and din=?'''
        set_args = [slim and slim.uuid(), component.time_spent, None,
                    1 if component.failed else 0]
        whe_args = [component.uuid(), component.uuid_train(), input_data.uuid()]
        self.query(sql, set_args + whe_args)
        print('Stored!')

    def data_exists(self, data):
        return self.get_data_by_uuid(data.uuid(), True) is not None

    def component_exists(self, component):
        return self.get_component(component, True) is not None

    def get_component(self, component, just_check_exists=False):
        field = 'arg'
        if just_check_exists:
            field = '1'
        self.query(f'select {field} from com where cid=?', [component.uuid()])
        result = self.get_one()
        if result is None:
            return None
        if just_check_exists:
            return True
        return result[field]

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

    def get_data_by_uuid(self, datauuid, just_check_exists=False):
        field = '1' if just_check_exists else 'des, bytes'
        self.query(f'''
            select 
                {field} 
            from 
                data 
                    left join name on name=nid 
                    left join dump on dumpd=duid
            where 
                did=?''', [datauuid])
        result = self.get_one()
        if result is None:
            return None
        if just_check_exists:
            return True
        falta
        pegar
        history
        return Data(name=result['des'], **unpack_data(result['bytes']))

    def get_data_by_name(self, name, fields=None, transformations=None,
                         just_check_exists=False):
        """
        To just recover an original classic dataset you can pass fields='X,y'
        (case insensitive).
        'None' means to recover as many fields as stored at the moment.
        When getting prediction data (i.e., results), it is better to specify
        which component made the predictions, otherwise a list of all found
        results is returned.
        :param name:
        :param fields: if None, get complet Data including predictions if any
        :param component:
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
