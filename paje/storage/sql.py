import warnings
from abc import abstractmethod

from paje.base.component import Component
from paje.base.data import Data
from paje.storage.storage import Cache
from paje.util.encoders import unpack_data, pack_comp, \
    uuid, zlibext_pack, zlibext_unpack, mysql_compress

# Disabling profiling when not needed.
try:
    import builtins

    profile = builtins.__dict__['profile']
except KeyError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


class SQL(Cache):
    @abstractmethod
    def _on_conflict(self, fields=None):
        pass

    @abstractmethod
    def _keylimit(self):
        pass

    @abstractmethod
    def _now_function(self):
        pass

    @abstractmethod
    def _auto_incr(self):
        pass

    @profile
    def _setup(self):
        print(self.name, 'creating tables...', self.info)

        # History of Data
        # ========================================================
        self.query(f'''
            create table if not exists hist (
                n integer NOT NULL primary key {self._auto_incr()},

                hid char(19) NOT NULL UNIQUE,

                txt LONGBLOB NOT NULL
            )''')

        # Columns of Data ======================================================
        self.query(f'''
            create table if not exists attr (
                n integer NOT NULL primary key {self._auto_incr()},

                aid char(19) NOT NULL UNIQUE,

                cols LONGBLOB NOT NULL
            )''')

        # Names of Data ========================================================
        self.query(f'''
            create table if not exists name (
                n integer NOT NULL primary key {self._auto_incr()},

                nid char(19) NOT NULL UNIQUE,

                des TEXT NOT NULL,

                attr char(19),

                FOREIGN KEY (attr) REFERENCES attr(aid)
            )''')
        self.query(f'CREATE INDEX nam0 ON name (des{self._keylimit()})')
        self.query(f'CREATE INDEX nam1 ON name (attr)')

        # Dump of Component instances  =========================================
        self.query(f'''
            create table if not exists inst (
                n integer NOT NULL primary key {self._auto_incr()},

                iid char(19) NOT NULL UNIQUE,
                dump LONGBLOB NOT NULL
            )''')

        # Logs for Component ===================================================
        self.query(f'''
            create table if not exists log (
                n integer NOT NULL primary key {self._auto_incr()},

                lid char(19) NOT NULL UNIQUE,

                msg TEXT NOT NULL,
                insl timestamp NOT NULL
            )''')
        self.query(f'CREATE INDEX log0 ON log (msg{self._keylimit()})')
        self.query(f'CREATE INDEX log1 ON log (insl)')

        # Components ===========================================================
        self.query(f'''
            create table if not exists com (
                n integer NOT NULL primary key {self._auto_incr()},

                cid char(19) NOT NULL UNIQUE,

                arg LONGBLOB NOT NULL,

                insc timestamp NOT NULL
            )''')
        self.query(f'CREATE INDEX com0 ON com (insc)')

        # Matrices/vectors
        # =============================================================
        self.query(f'''
            create table if not exists mat (
                n integer NOT NULL primary key {self._auto_incr()},

                mid char(19) NOT NULL UNIQUE,

                w integer,
                h integer,

                val LONGBLOB NOT NULL                 
            )''')
        self.query(f'CREATE INDEX mat0 ON mat (w)')
        self.query(f'CREATE INDEX mat1 ON mat (h)')

        # Datasets =============================================================
        self.query(f'''
            create table if not exists data (
                n integer NOT NULL primary key {self._auto_incr()},

                did char(19) NOT NULL UNIQUE,

                name char(19) NOT NULL,
                hist char(19) NOT NULL,

                x char(19),
                y char(19),

                z char(19),
                p char(19),

                u char(19),
                v char(19),

                w char(19),
                q char(19),

                r char(19),
                s char(19),

                t char(19),

                l char(19),
                m char(19),

                insd timestamp NOT NULL,
                upd timestamp,

                unique(name, hist),
                FOREIGN KEY (name) REFERENCES name(nid),
                FOREIGN KEY (hist) REFERENCES hist(hid),
                
                FOREIGN KEY (x) REFERENCES mat(mid),
                FOREIGN KEY (y) REFERENCES mat(mid),
                FOREIGN KEY (z) REFERENCES mat(mid),
                FOREIGN KEY (p) REFERENCES mat(mid),
                FOREIGN KEY (u) REFERENCES mat(mid),
                FOREIGN KEY (v) REFERENCES mat(mid),
                FOREIGN KEY (w) REFERENCES mat(mid),
                FOREIGN KEY (q) REFERENCES mat(mid),
                FOREIGN KEY (r) REFERENCES mat(mid),
                FOREIGN KEY (s) REFERENCES mat(mid),
                FOREIGN KEY (t) REFERENCES mat(mid),
                FOREIGN KEY (l) REFERENCES mat(mid),
                FOREIGN KEY (m) REFERENCES mat(mid)
               )''')
        # guardar last comp nao adianta pq o msm comp pode ser aplicado
        # varias vezes
        # history não vai conter comps inuteis como pipes e switches, apenas
        # quem transforma Data, ou seja, faz updated().
        self.query(f'CREATE INDEX data0 ON data (insd)')
        self.query(f'CREATE INDEX data1 ON data (name)')  # needed on FKs?
        self.query(f'CREATE INDEX data2 ON data (hist)')
        self.query(f'CREATE INDEX data3 ON data (upd)')
        self.query(f'CREATE INDEX datax ON data (x)')
        self.query(f'CREATE INDEX datay ON data (y)')
        self.query(f'CREATE INDEX dataz ON data (z)')
        self.query(f'CREATE INDEX datap ON data (p)')
        self.query(f'CREATE INDEX datau ON data (u)')
        self.query(f'CREATE INDEX datav ON data (v)')
        self.query(f'CREATE INDEX dataw ON data (w)')
        self.query(f'CREATE INDEX dataq ON data (q)')
        self.query(f'CREATE INDEX datar ON data (r)')
        self.query(f'CREATE INDEX datas ON data (s)')
        self.query(f'CREATE INDEX datat ON data (t)')
        self.query(f'CREATE INDEX datal ON data (l)')
        self.query(f'CREATE INDEX datam ON data (m)')

        # Results ==============================================================
        self.query(f'''
            create table if not exists res (
                n integer NOT NULL primary key {self._auto_incr()},

                node varchar(19) NOT NULL,

                com char(19) NOT NULL,
                op char(1) NOT NULL,

                dtr char(19) NOT NULL,
                din char(19) NOT NULL,

                log char(19),

                dout char(19),
                spent FLOAT,
                inst char(19),

                fail TINYINT,

                start TIMESTAMP NOT NULL,
                end TIMESTAMP NOT NULL,
                alive TIMESTAMP NOT NULL,

                tries int NOT NULL,
                locks int NOT NULL,
                mark varchar(256),

                UNIQUE(com, op, dtr, din),
                FOREIGN KEY (com) REFERENCES com(cid),
                FOREIGN KEY (dtr) REFERENCES data(did),
                FOREIGN KEY (din) REFERENCES data(did),
                FOREIGN KEY (dout) REFERENCES data(did),
                FOREIGN KEY (inst) REFERENCES inst(iid),
                FOREIGN KEY (log) REFERENCES log(lid)
            )''')
        self.query('CREATE INDEX res0 ON res (dout)')
        self.query('CREATE INDEX res1 ON res (spent)')
        self.query('CREATE INDEX res2 ON res (inst)')
        self.query('CREATE INDEX res3 ON res (fail)')
        self.query('CREATE INDEX res4 ON res (start)')
        self.query('CREATE INDEX res5 ON res (end)')
        self.query('CREATE INDEX res6 ON res (alive)')
        self.query('CREATE INDEX res7 ON res (node)')
        self.query('CREATE INDEX res8 ON res (tries)')
        self.query('CREATE INDEX res9 ON res (locks)')
        self.query('CREATE INDEX res10 ON res (log)')
        self.query(f'CREATE INDEX res11 ON res (mark{self._keylimit()})')
        self.query(f'CREATE INDEX res12 ON res (op)')
        # TODO: remove all FK constraints to speedup sqlite? 10% of improvement?

    @profile
    def get_one(self) -> dict:
        """
        Get a single result after a query, no more than that.
        :return:
        """
        row = self.cursor.fetchone()
        if row is None:
            return None
        row2 = self.cursor.fetchone()
        if row2 is not None:
            print(self.name, 'first row', row)
            while row2:
                print(self.name, 'extra row', row2)
                row2 = self.cursor.fetchone()
            raise Exception(self.name + '  Excess of rows')
        return dict(row)

    @profile
    def get_all(self) -> list:
        """
        Get a list of results after a query.
        :return:
        """
        rows = self.cursor.fetchall()
        # if len(rows) == 0:
        #     return None
        return [dict(row) for row in rows]

    @profile
    def store_matvec(self, uuid, dump, w, h):
        """
        Store the given pair uuid-dump of a matrix/vector.
        :type uuid:
        :param dump:
        :param w: width
        :param h: height
        :return:
        """
        sql = f'''
            insert or ignore into mat values (
            null,
            ?,
            ?, ?,
            ?)
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [uuid, w, h, dump])

    @profile
    def store_data_impl(self, data: Data):
        """
        Check if the given data was already stored before,
        and complete with the provided fields as needed.
        The sequence of queries is planned to minimize traffic and CPU load,
        otherwise it would suffice to just send 'insert or ignore' of dumps.
        :param data: Data
        :return:
        """
        # Check if Data already exists and which fields are already stored.
        self.query(f'''
                    select * from data
                    where did=?''', [data.uuid()])
        rone = self.get_one()

        if rone is None:
            # Check if dumps of matrices/vectors already exist (improbable).
            uuid_dump = data.uuids_dumps()
            qmarks = ','.join(['?'] * len(uuid_dump))
            self.query(f'''
                        select mid from mat
                        where mid in ({qmarks})''', list(uuid_dump.keys()))
            rall = self.get_all()
            mids = [row['mid'] for row in rall]
            # print('res getall (check None here?)', type(rall), rall, mids)

            # Insert only dumps that are missing in storage
            dumps2store = {k: v for k, v in uuid_dump.items()
                           if k not in mids}
            uuid_field = data.uuids_fields()
            for uuid_, dump in dumps2store.items():
                self.store_matvec(
                    uuid_, dump, data.width(uuid_field[uuid_]),
                    data.height(uuid_field[uuid_])
                )

            # Create metadata for upcoming row at table 'data'.
            self.store_metadata(data)

            # Create row at table 'data'. ---------------------
            sql = f''' 
                insert into data values (
                    NULL,
                    ?,
                    ?, ?,
                    ?,?,
                    ?,?,
                    ?,?,
                    ?,?,
                    ?,?,
                    ?,
                    ?,?,
                    {self._now_function()},
                    null
                )
                '''
            data_args = [data.uuid(),
                         data.name_uuid(), data.history_uuid(),
                         data.field_uuid('x'), data.field_uuid('y'),
                         data.field_uuid('z'), data.field_uuid('p'),
                         data.field_uuid('u'), data.field_uuid('v'),
                         data.field_uuid('w'), data.field_uuid('q'),
                         data.field_uuid('r'), data.field_uuid('s'),
                         data.field_uuid('t'),
                         data.field_uuid('l'), data.field_uuid('m')
                         ]
            from sqlite3 import IntegrityError as IntegrityErrorSQLite
            from pymysql import IntegrityError as IntegrityErrorMySQL
            try:
                self.query(sql, data_args)
                # unfortunately,
                # it seems that FKs generate the same exception as reinsertion.
                # so, missing FKs will might not be detected here.
                # not a worrying issue whatsoever.
            except IntegrityErrorSQLite as e:
                print(self.name,
                      f'Unexpected: Data already stored before!', data.uuid())
            except IntegrityErrorMySQL as e:
                print(self.name,
                      f'Unexpected: Data already stored before!', data.uuid())
            else:
                print(self.name, f': Data inserted', data.name())

        else:
            if self.debug:
                print('Check if data comes with new matrices/vectors '
                      '(improbable).')
            stored_dumps = {k: v for k, v in rone.items() if v is not None}
            fields_low = [k.lower() for k in data.matvecs().keys()]
            fields2store = [f for f in fields_low if f
                            is not None and f not in stored_dumps]

            if self.debug:
                print('Insert only dumps that are missing in storage')
            dumps2store = {data.field_uuid(f): (data.field_dump(f), f)
                           for f in fields2store}
            to_update = {}
            for uuid_, (dump, field) in dumps2store:
                self.store_matvec(uuid_, dump, data.width(field),
                                  data.height(field))
                to_update[field] = uuid_

            if self.debug:
                print('Update row at table "data" if needed...')
            if len(to_update) > 0:
                sql = f''' 
                    update data set
                        {','.join([f'{k}=?' for k in to_update.keys()])}
                        insd=insd,
                        upd={self._now_function()}
                    where
                        did=?
                    '''
                self.query(sql, list(to_update.values()) + [data.uuid()])
                print(self.name, f': Data updated', data.name())

    @profile
    def store_metadata(self, data: Data):
        """
        Intended to be used before Data is stored.
        :param data:
        :return:
        """
        # attr ---------------------------------------------------------
        # TODO: avoid sending long cols blob when unneeded
        cols = zlibext_pack(data.columns())
        uuid_cols = uuid(cols)
        sql = f'''
            insert or ignore into attr values (
                NULL,
                ?,
                ?
            );'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [uuid_cols, cols])

        # name ---------------------------------------------------------
        # TODO: avoid sending long names when unneeded
        sql = f'''
            insert or ignore into name values (
                NULL,
                ?,
                ?,
                ?
            );'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [data.name_uuid(), data.name(), uuid_cols])

        # history ------------------------------------------------------
        # TODO: avoid sending long hist blob when unneeded
        sql = f'''
            insert or ignore into hist values (
                NULL,
                ?,
                ?
            )'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [data.history_uuid(), zlibext_pack(data.history())])
            # TODO: codify a custom zipper for history to be shorter
            # TODO: create a lazy method hist_dump() in Data

    @profile
    def lock_impl(self, component, op, input_data):
        """
        Store 'input_data' and 'component' if they are not yet stored.
        Insert a locking row that corresponds to comp,op,train_data,input_data.
        'op'eration should be 'a'pply or 'u'se
        :param component:
        :param op: operation to store and also for logging purposes
        :param input_data:
        :return:
        """
        if self.debug:
            print(self.name, 'Locking...', op)

        # Store input set if not existent yet.
        # Calling impl to avoid nested storage doing the same twice.
        self.store_data_impl(input_data)

        # Store component (if not existent yet) and attempt to acquire lock.
        # Mark as locked_by_others otherwise.
        nf = self._now_function()
        sql = f'''
            insert or ignore into com values (
                NULL,
                ?,
                ?,
                {nf}
            )'''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.query(sql, [component._uuid,
                             mysql_compress(component.serialized().encode())])

        sql = f'''insert into res values (
                null,
                ?,
                ?, ?,
                ?, ?,
                null,
                null, null, null,
                null,
                {nf}, '0000-00-00 00:00:00', '0000-00-00 00:00:00',
                0, 0, null
            )'''
        args_res = [self.hostname,
                    component._uuid, op,
                    component.train_data_uuid__mutable(), input_data._uuid]
        from sqlite3 import IntegrityError as IntegrityErrorSQLite
        from pymysql import IntegrityError as IntegrityErrorMySQL
        try:
            self.query(sql, args_res)
        except IntegrityErrorSQLite as e:
            print(self.name,
                  f'Unexpected lock! Giving up my turn on {op} ppy/se', e)
            component.locked_by_others = True
        except IntegrityErrorMySQL as e:
            print(self.name,
                  f'Unexpected lock! Giving up my turn on {op} ppy/se', e)
            component.locked_by_others = True
        else:
            component.locked_by_others = False
            print(self.name,
                  f'Now locked for {op}[ppying/sing] {component.name}')

    @profile
    def get_result_impl(self, compo: Component, op, input_data):
        """
        Look for a result in database. Download only affected matrices/vectors.
        ps.: put a model inside component if it is 'dump_it'-enabled
        :return: Resulting Data
        """
        if compo.failed or compo.locked_by_others:
            return None, True, compo.failed is not None
        fields = compo.modifies(op)

        if compo._dump_it:
            raise Exception('Are we really starting to store dump of '
                            'components?')
        self.query(f'''
            select 
                des, spent, fail, end, node, txt as history, cols
                {',' + ','.join(fields) if len(fields) > 0 else ''}
                {', dump' if compo._dump_it else ''}
            from 
                res 
                    left join data on dout = did
                    left join name on name = nid
                    left join hist on hist = hid
                    left join attr on attr = aid
                    {'left join inst on inst = iid' if compo._dump_it else 
        ''}                    
            where                
                com=? and op=? and dtr=? and din=?''',
                   [compo._uuid,
                    op,
                    compo.train_data_uuid__mutable(),
                    input_data.uuid()])
        result = self.get_one()
        if result is None:
            return None, False, False
        if result['des'] is not None:
            # sanity check
            if result['des'] != input_data.name():
                raise Exception('Result name differs from input data',
                                f"{result['des']}!={input_data.name()}")

            # Recover relevant matrices/vectors.
            dic = {}
            for field in fields:
                mid = result[field]
                if mid is not None:
                    self.query(f'select val,w,h from mat where mid=?', [mid])
                    rone = self.get_one()
                    if rone is not None:
                        dic[Data.fields_in_constructor_format[field]] = \
                            unpack_data(rone['val'], rone['w'], rone['h'])

            # Create Data.
            history = zlibext_unpack(result['history'])
            columns = zlibext_unpack(result['cols'])
            data = Data(name=result['des'], history=history, columns=columns,
                        **dic)

            # Join untouched matrices/vectors.
            output_data = input_data.merged(data)
        else:
            output_data = None
        compo.model = result['dump'].model if 'dump' in result else None
        compo.time_spent = result['spent']
        compo.failed = result['fail'] and result['fail'] == 1
        compo.locked_by_others = result['end'] == '0000-00-00 00:00:00'
        compo.node = result['node']
        return output_data, True, compo.failed is not None

    @profile
    def store_result_impl(self, component, op, input_data, output_data):
        """
        Store a result and remove lock.
        :param component:
        :param op:
        :param input_data:
        :param output_data:
        :return:
        """

        # Store resulting Data
        if output_data is not None:
            # We call impl here,
            # to avoid nested storage trying to do the same work twice.
            self.store_data_impl(output_data)

        # Remove lock and point result to data inserted above.
        # We should set all timestamp fields even if with the same old value,
        # due to automatic updates by DBMSs.
        # Data train was inserted and dtr was created when locking().
        now = self._now_function()

        # Store dump if requested.
        dump_uuid = component._dump_it and uuid(
            (component._uuid + op +
             component.train_data_uuid__mutable() + input_data.uuid()).encode()
        )
        if component._dump_it:
            sql = f'insert or ignore into inst values (null, ?, ?)'
            # pack_comp is nondeterministic and its result is big,
            # so we need to identify it by other, deterministic, means
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.query(sql, [dump_uuid, pack_comp(component)])

        # Store a log if apply() failed.
        log_uuid = component.failure and uuid(component.failure.encode())
        if component.failure is not None:
            sql = f'insert or ignore into log values (null, ?, ?, {now})'
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.query(sql, [log_uuid, component.failure])

        # Unlock and save result.
        fail = 1 if component.failed else 0
        sql = f'''
                update res set 
                    log=?,
                    dout=?, spent=?, inst=?,
                    fail=?,
                    start=start, end={now}, alive={now},
                    mark=?
                where
                    com=? and op=? and 
                    dtr=? and din=?
                '''
        set_args = [log_uuid, output_data and output_data.uuid(),
                    component.time_spent, dump_uuid, fail, component.mark]
        where_args = [component._uuid, op,
                      component.train_data_uuid__mutable(), input_data.uuid()]
        # TODO: is there any important exception to handle here?
        self.query(sql, set_args + where_args)
        print(self.name, 'Stored!\n')

    @profile
    def get_data_by_name_impl(self, name, fields=None, history=None):
        """
        To just recover the original dataset you can pass history=None.
        Specify fields if you want to reduce traffic, otherwise all available
        fields will be fetched.

        ps. 1: Obviously, when getting prediction data (i.e., results),
         the history which led to the predictions should be provided.
        :param name:
        :param fields: None=get full Data; case insensitive; e.g. 'X,y,Z'
        :param history: list of JSON(TODO:json??) strings describing components
        :param just_check_exists:
        :return:
        """
        if history is None:
            history = []
        hist_uuid = uuid(zlibext_pack(history))

        sql = f'''
                select 
                    x,y,z,p,u,v,w,q,r,s,t,l,m,cols,des
                from 
                    data 
                        left join name on name=nid 
                        left join attr on attr=aid
                where 
                    des=? and hist=?'''
        self.query(sql, [name, hist_uuid])
        row = self.get_one()
        if row is None:
            return None

        # Recover requested matrices/vectors.
        dic = {'name': name, 'history': history}
        if fields is None:
            flst = [k for k, v in row.items() if len(k) == 1 and v is not None]
        else:
            flst = fields.split(',')
        for field in Data.list_to_case_sensitive(flst):
            mid = row[field.lower()]
            if mid is not None:
                self.query(f'select val,w,h from mat where mid=?', [mid])
                rone = self.get_one()
                dic[field] = unpack_data(rone['val'], rone['w'], rone['h'])
        return Data(columns=zlibext_unpack(row['cols']), **dic)

    # def get_component_by_uuid(self, component_uuid, just_check_exists=False):
    #     field = 'arg'
    #     if just_check_exists:
    #         field = '1'
    #     self.query(f'select {field} from com where cid=?', [component_uuid])
    #     result = self.get_one()
    #     if result is None:
    #         return None
    #     if just_check_exists:
    #         return True
    #     return result[field]

    # def get_component(self, component, train_data, input_data,
    #                   just_check_exists=False):
    #     field = 'bytes,arg'
    #     if just_check_exists:
    #         field = '1'
    #     self.query(f'select {field} '
    #                f'from res'
    #                f'   left join dump on dumpc=duic '
    #                f'   left join com on com=cid '
    #                f'where cid=? and dtr=? and din=?',
    #                [component.uuid(),
    #                 component.train_data.uuid(),
    #                 input_data.uuid()])
    #     result = self.get_one()
    #     if result is None:
    #         return None
    #     if just_check_exists:
    #         return True
    #     return Component.resurrect_from_dump(result['bytes'],
    #                                          **json_unpack(result['arg']))

    @profile
    def get_data_by_uuid_impl(self, datauuid):
        sql = f'''
                select 
                    x,y,z,p,u,v,w,q,r,s,t,l,m,cols,txt,des
                from 
                    data 
                        left join name on name=nid 
                        left join hist on hist=hid
                        left join attr on attr=aid
                where 
                    did=?'''
        self.query(sql, [datauuid])
        row = self.get_one()
        if row is None:
            return None

        # Recover requested matrices/vectors.
        # TODO: surely there is duplicated code to be refactored in this file!
        dic = {'name': row['des'],
               'history': zlibext_unpack(row['txt'])}
        fields = [k for k, v in row.items() if len(k) == 1 and v is not None]
        for field in Data.list_to_case_sensitive(fields):
            mid = row[field.lower()]
            if mid is not None:
                self.query(f'select val,w,h from mat where mid=?', [mid])
                rone = self.get_one()
                dic[field] = unpack_data(rone['val'], rone['w'], rone['h'])
        return Data(columns=zlibext_unpack(row['cols']), **dic)

    @profile
    def get_finished_names_by_mark_impl(self, mark):
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
                    des, txt
                from
                    res join data on dtr=did 
                        join name on name=nid 
                        join hist on hist=hid 
                where
                    end!='0000-00-00 00:00:00' and 
                    fail=0 and mark=?
            """, [mark])
        rows = self.get_all()
        if rows is None:
            return None
        else:
            for row in rows:
                row['name'] = row.pop('des')
                row['history'] = zlibext_unpack(row.pop('txt'))
            return rows

    @profile
    def query(self, sql, args=None):
        if self.read_only and not sql.startswith('select '):
            print(self.name, '========================================\n',
                  'Attempt to write onto read-only storage!', sql)
            self.cursor.execute('select 1')
            return
        if args is None:
            args = []
        from paje.storage.mysql import MySQL
        msg = self._interpolate(sql, args)
        if self.debug:
            print(self.name, msg)
        if isinstance(self, MySQL):
            sql = sql.replace('?', '%s')
            sql = sql.replace('insert or ignore', 'insert ignore')
            # self.connection.ping(reconnect=True)

        try:
            self.cursor.execute(sql, args)
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

    # def _data_exists(self, data):
    #     return self.get_data_by_uuid(data.uuid(), True) is not None

    # def _component_exists(self, component):
    #     return self.get_component_by_uuid(component.uuid(), True) is not None

    @profile
    def __del__(self):
        try:
            self.connection.close()
        except Exception as e:
            # print('Couldn\'t close database, but that\'s ok...', e)
            pass

    @staticmethod
    @profile
    def _interpolate(sql, lst0):
        lst = [str(w)[:100] for w in lst0]
        zipped = zip(sql.replace('?', '"?"').split('?'), map(str, lst + ['']))
        return ''.join(list(sum(zipped, ()))).replace('"None"', 'NULL')

        # # upsert, works for mysql and sqlite 3.24 (not yet in python 3.7)
        # sql = f'''
        #     insert into data values (
        #         NULL,
        #         ?,
        #         ?, ?,
        #         ?,?,
        #         ?,?,
        #         ?,?,
        #         ?,?,
        #         ?,?,
        #         ?,
        #         {self._now_function()},
        #         null
        #     )
        #     {self._on_conflict()}
        #     x = coalesce(values(x), x),
        #     y = coalesce(values(y), y),
        #     z = coalesce(values(z), z),
        #     p = coalesce(values(p), p),
        #     u = coalesce(values(u), u),
        #     v = coalesce(values(v), v),
        #     w = coalesce(values(w), w),
        #     q = coalesce(values(q), q),
        #     r = coalesce(values(r), r),
        #     s = coalesce(values(s), s),
        #     t = coalesce(values(t), t),
        #     insd = insd,
        #     upd = {self._now_function()}
        #     '''
