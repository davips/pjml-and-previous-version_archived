import warnings
from abc import abstractmethod
from sqlite3 import IntegrityError as IntegrityErrorSQLite

from paje.base.data import Data
from paje.result.storage import Cache
from pymysql import IntegrityError as IntegrityErrorMySQL

from paje.util.encoders import unpack_data


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

                txt TEXT NOT NULL
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
                typ varchar(20),
                bytes LONGBLOB NOT NULL
            )''')
        self.query(f'CREATE INDEX dump0 ON dump (typ)')

        # Logs for Component ===================================================
        self.query(f'''
            create table if not exists log (
                n integer NOT NULL primary key {self.auto_incr()},

                lid char(19) NOT NULL UNIQUE,

                msg TEXT NOT NULL,
                insl timestamp NOT NULL
            )''')
        self.query(f'CREATE INDEX log0 ON log (msg{self.keylimit()})')
        self.query(f'CREATE INDEX log1 ON log (insl)')

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
        # guardar last comp nao adianta pq o msm comp pode ser aplicado
        # varias vezes
        # history não vai conter comps inuteis como pipes e switches, apenas
        # quem transforma Data, ou seja, faz updated().
        self.query(f'CREATE INDEX data0 ON data (shape{self.keylimit()})')
        self.query(f'CREATE INDEX data1 ON data (insd)')
        self.query(f'CREATE INDEX data2 ON data (dumpd)')
        self.query(f'CREATE INDEX data3 ON data (name)')  # needed?
        self.query(f'CREATE INDEX data4 ON data (fields)')  # needed?
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
                mark varchar(256),

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
        self.query('CREATE INDEX res11 ON res (mark)')

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
            input_data.shrink_to(component.fields_to_keep_after_use()))

        # Store component (if inexistent yet) and attempt to acquire lock.
        # Mark as locked_by_others otherwise.
        nf = self.now_function()
        sql = f'''
            insert or ignore into com values (
                NULL,
                ?,
                ?,
                {nf}
            )'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [component.uuid(), component.serialized()])

        sql = f'''insert into res values (
                null,
                ?,
                ?, ?, ?, null,
                null, null, null,
                null,
                {nf}, '0000-00-00 00:00:00', '0000-00-00 00:00:00',
                0, 0, null
            )'''
        args_res = [self.hostname,
                    component.uuid(), component.train_data__mutable().uuid(),
                    input_data.uuid()]
        try:
            self.query(sql, args_res)
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

    def get_result(self, component, input_data):
        """
        Look for a result in database.
        ps.: put a model inside component if it is 'dump_it'-enabled
        :return: Resulting Data
        """
        if component.failed or component.locked_by_others:
            return None
        self.query(f'''
            select 
                des, bytes, spent, fail, end, node, txt as history, cols
                {', duc.bytes as model' if component.dump_it else ''}
            from 
                res 
                    left join data on dout = did
                    left join dump on dumpd = duid
                    left join name on name = nid
                    left join hist on hist = hid
                    {'left join dump on dumpc = duid'
        if component.dump_it else ''}                    
            where                
                com=? and dtr=? and din=?''',
                   [component.uuid(),
                    component.train_data__mutable().uuid(),
                    input_data.uuid()])
        result = self.get_one()
        if result is None:
            return None
        if 'bytes' in result and result['bytes']:
            if result['des'] != input_data.name():
                raise Exception('Result name differs from input data',
                                f"{result['des']}!={input_data.name()}")
            data = Data(name=result['des'],
                        history=result['history'].split('|'),
                        **unpack_data(result['bytes']),
                        columns=result['cols'].split('|'))
            # print(input_data.shapes(), len(input_data.history()), \
            #       input_data.history())
            # print(data.shapes(), len(data.history()), data.history())
            output_data = input_data.shrink_to(
                component.fields_to_keep_after_use()
            ).merged(data)
            # print(output_data.shapes(), len(output_data.history()), \
            #       output_data.history())

        else:
            output_data = None
        component._model_dump = result['model'] if 'model' in result else None
        component.time_spent = result['spent']
        component.failed = result['fail'] and result['fail'] == 1
        component.locked_by_others = result['end'] == '0000-00-00 00:00:00'
        component.node = result['node']
        return output_data

    def store_data(self, data: Data):
        # Check first with a low cost query if data already exists.
        if self.data_exists(data):
            if self.debug:
                print('Data already exists:' + data.uuid(), data.name())
            return
        else:
            print('Storing...', data.name(), data.uuid())

        # Insert dump of data and data info.
        # Catch exception in the event of another job winning the race.
        # Dumps can have duplicate, e.g. when two models give the same
        # predictions (usually in the same dataset), so 'insert or ignore'.
        # In the case of training data (e.g. only X,y) the same Data can be
        # inserted twice by different components, but this is checked above
        # by data_exists(). If in the meantime, while the interpreter is
        # reading this comment, someone else inserts just the same Data,
        # we will have to handle it as an exception.
        # ps. : in the presence of predictions (z),
        # UUID will change according to the new set of fields,
        # otherwise, it will depend only on 'Data.name' and 'Data.fields'.

        sql = f'''
            insert or ignore into dump values (
                null,
                ?, 'data',
                ?
            )'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [data.dump_uuid(), data.dump()])

        sql = f'''
            insert or ignore into name values (
                NULL,
                ?,
                ?,
                ?
            );'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [data.name_uuid(), data.name(),
                             '|'.join(data.columns())])

        sql = f'''
            insert or ignore into hist values (
                NULL,
                ?,
                ?
            )'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [data.history_uuid(), '|'.join(data.history())])

        sql = f'''
            insert into data values (
                NULL,
                ?,
                ?, ?, ?,
                ?, ?,
                {self.now_function()}
            );'''
        data_args = [data.uuid(),
                     data.name_uuid(), data.fields(), data.history_uuid(),
                     data.dump_uuid(),
                     '|'.join([k + ':' + str(v)
                               for k, v in data.shapes().items()])]
        try:
            self.query(sql, data_args)
        except IntegrityErrorSQLite as e:
            print(f'Data already store before!', data.uuid())
        except IntegrityErrorMySQL as e:
            print(f'Data already store before!', data.uuid())
        else:
            print(f'Data inserted', data.uuid())

    def store(self, component, input_data, output_data):
        """
        Store a result and remove lock.
        :param component:
        :param input_data:
        :param output_data:
        :return:
        """

        # Store resulting Data
        slim = output_data and output_data.shrink_to(
            component.fields_to_store_after_use())
        slim and self.store_data(slim)

        # Remove lock and point result to data inserted above.
        # TODO: is there any important exception to handle here?
        now = self.now_function()
        model_dump = component.dump_it and component.model_dump()
        # We should set all timestamp fields even if with the same old value.
        # Data train inserted and dtr was created when locking().
        # 'or ignore' because different models can have the same predictions.
        sql = f'''insert or ignore into dump values (
                    null,
                    ?, 'model', ?
                )'''
        dump_args = [component.model_uuid(), model_dump]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, dump_args)

        sql = f'''update res set 
                    dout=?, spent=?,
                    dumpc=?,
                    fail=?,
                    start=start, end={now}, alive={now},
                    mark=?
                where
                    com=? and dtr=? and din=?
                '''
        resargs1 = [
            slim and slim.uuid(), component.time_spent,  # dout, spent
            component.model_uuid(),  # dumpc
            1 if component.failed else 0,  # fail
            None if component.train_data__mutable().uuid() == input_data.uuid()
            else component.mark]  # mark
        resargs2 = [component.uuid(), component.train_data__mutable().uuid(),
                    input_data.uuid()]  # com, dtr, din
        self.query(sql, resargs1 + resargs2)

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
    def query(self, sql, args=None, commit=True):
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

        try:
            self.cursor.execute(sql, args)
            if commit:
                self.connection.commit()
        except Exception as ex:
            # From a StackOverflow answer...
            import sys
            import traceback
            msg = self.info + '\n' + msg
            # Gather the information from the original exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            # Format the original exception for a nice printout:
            traceback_string = ''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback))
            # Re-raise a new exception of the same class as the original one
            raise type(ex)(
                "%s\norig. trac.:\n%s\n" % (msg, traceback_string))

    def get_data_by_uuid(self, datauuid, just_check_exists=False):
        sql_fields = '1' if just_check_exists else 'des, cols, bytes, txt'
        self.query(f'''
                select 
                    {sql_fields} 
                from 
                    data 
                        left join name on name=nid 
                        left join dump on dumpd=duid
                        left join hist on hist=hid
                where 
                    did=?''', [datauuid])
        result = self.get_one()
        if result is None:
            return None
        if just_check_exists:
            return True
        return Data(name=result['des'],
                    columns=result['cols'].split('|'),
                    history=result['history'].split('|'),
                    **unpack_data(result['bytes']))

    def get_data_by_name(self, name, fields=None, history=None,
                         just_check_exists=False):
        """
        To just recover the original dataset you can pass history=None.
        Specify fields as needed, otherwise all fields will be merged into
        a single Data - in this case, metadata of the first Data will prevail.

        When getting prediction data (i.e., results),
         the history which led to the predictions should be provided.
        :param name:
        :param fields: None=get full Data; case insensitive; e.g. 'X,y,Z'
        :param history: list of JSON strings describing components
        :param just_check_exists:
        :return:
        """
        history = '' if history is None else '|'.join(history)
        sql_fields = '1' if just_check_exists else 'cols, bytes, txt'

        sql = f'''
                select 
                    {sql_fields} 
                from 
                    data 
                        left join name on name=nid 
                        left join dump on dumpd=duid
                        left join hist on hist=hid
                where 
                    des=? {f'and fields=?' if fields else ''} and
                    txt=?'''
        args = [name, fields.upper(), history] if fields else [name, history]

        self.query(sql, args)
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        if just_check_exists:
            return True

        # Merge all recovered datas into one.
        dic = {}
        for row in rows:
            dic.update(unpack_data(row['bytes']))

        return just_check_exists or Data(name=name, history=history,
                                         columns=rows[0]['cols'].split('|'),
                                         **unpack_data(rows[0]['bytes']))

    def count_results(self, component, data):
        """
        Useless method.
        :param component:
        :param data:
        :return:
        """
        self.query(f'select n from res where com=? and dtr=?',
                   [component.uuid(), data.uuid()])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return 0
        else:
            return len(rows)

    def get_finished_names_by_mark(self, mark):
        """
        Finished means nonfailed and unlocked results.
        The original datasets will not be returned.
        original dataset = stored with no history of transformations.
        All results are returned (apply & use, as many as there are),
         so the names come along with their history,
        :param mark:
        :return: [dicts]
        """
        self.query(f"""
                select
                    des, txt as history
                from
                    res join data on dtr=did 
                        join name on name=nid 
                        join hist on hist=hid 
                where
                    end!='0000-00-00 00:00:00' and 
                    fail=0 and mark=?
            """, [mark])
        rows = self.cursor.fetchall()
        if rows is None or len(rows) == 0:
            return None
        else:
            for row in rows:
                row['name'] = row.pop('des')
            return rows

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
