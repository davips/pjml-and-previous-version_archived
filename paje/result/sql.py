from paje.base.data import Data
from paje.result.storage import Cache, unpack, pack


class SQL(Cache):
    def setup(self):
        if self.debug:
            print('creating tables...')
        self.query("create table if not exists result "
                   "(idcomp varchar(32), idtrain varchar(32), "
                   "idtest varchar(32), "
                   "trainout LONGBLOB, testout LONGBLOB, "
                   "timetrain FLOAT, timetest FLOAT, "
                   "dump LONGBLOB, failed BOOLEAN, PRIMARY KEY("
                   "idcomp, idtrain, idtest))")
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

    def got(self):
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

    def result_exists(self, component, train, test):
        return self.get_result(component, train, test, True) != (None, None,
                                                                 None)

    def component_exists(self, component):
        return self.get_component(component, True) is not None

    def data_exists(self, data):
        return data is None or self.get_data(data, True) is not None

    def get_result(self, component, train, test,
                   just_check_exists=False, fields_to_store=None):
        """
        Look for a result in database.
        :param just_check_exists:
        :param component:
        :param train:
        :param test:
        :param fields_to_store:
        :return:
        """
        if fields_to_store is None:
            fields_to_store = []
        fields = 'trainout, testout, failed'
        if just_check_exists:
            fields = '1'
        self.query(
            f"select {fields} from result where "
            "idcomp=? and idtrain=? and idtest=?",
            [component.uuid(), train.uuid, (test or 0) and test.uuid])
        res = self.got()
        if res is None:
            return None, None, None
        else:
            if just_check_exists:
                return True, True, True
            trainout = train.updated(**unpack(res[0]).select(fields_to_store))
            testout = test and test.updated(**unpack(res[1]).select(
                fields_to_store))
            return trainout, testout, res[2]

    def get_component(self, component, just_check_exists=False):
        field = 'dic'
        if just_check_exists:
            field = '1'
        self.query(f'select {field} from args where idcomp=?',
                   [component.uuid()])
        res = self.got()
        if res is None:
            return None
        else:
            return res[0]

    def get_data(self, data, just_check_exists=False):
        field = 'data'
        if just_check_exists:
            field = '1'
        self.query(f'select {field} from dset where iddset=?', [data.uuid])
        res = self.got()
        if res is None:
            return None, None
        else:
            return Data(**unpack(res[0]))

    def get_component_dump(self, component, train, test,
                           just_check_exists=False):
        raise NotImplementedError('get model')

    def store(self, component, train, test, trainout, testout,
              time_spent_tr, time_spent_ts, fields_to_store):
        slim_trainout = trainout and trainout.sub(fields_to_store)
        slim_testout = testout and testout.sub(fields_to_store)
        if not self.result_exists(component, train, test):
            # try:
            #     dump = pack(component)
            # except:
            # # except MemoryError as error:
            #     component.warning('Aborting dump storing due to memory issues.')
            #     from paje.module.modelling.classifier.nb import NB
            # TODO: dumps are not saved anymore!
            dump = None
            self.query("insert into result values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       [component.uuid(), train.uuid,
                        (test or 0) and test.uuid,
                        pack(slim_trainout), pack(slim_testout),
                        time_spent_tr, time_spent_ts,
                        dump, component.failed])
        if not self.component_exists(component):
            self.query("insert into args values (?, ?)",
                       [component.uuid(), component.serialized()])
        if not self.data_exists(train):
            self.query("insert into dset values (?, ?)", [train.uuid,
                                                          pack(train)])
        if not self.data_exists(test):
            self.query("insert into dset values (?, ?)",
                       [test.uuid, pack(test)])
        self.connection.commit()

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

    def __del__(self):
        self.connection.close()
