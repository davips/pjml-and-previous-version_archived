import warnings
from pathlib import Path

import _pickle as pickle

from paje.base.component import Component
from paje.base.data import Data
from paje.storage.cache import Cache
from paje.util.encoders import uuid


class PickledFile(Cache):
    @staticmethod
    def _outdata_uuid(component, input_data):
        return uuid(
            (uuid(input_data.name.encode()) + (
                component.config, component.op,
                input_data.history
            )).encode())

    def store_data_impl(self, data):
        pickle.dump(data, data.uuid() + '.pickle')

    def lock_impl(self, component, input_data):
        Path(self._outdata_uuid(component, input_data) + '.pickle').touch()

    def get_result_impl(self, component, input_data):
        # Failed?
        if Path(self._outdata_uuid(component, input_data) + '.fail'):
            component.failed = True
            ended = component.failed is not None
            return None, True, ended
        else:
            ended = component.failed is not None

        # Not started yet?
        if not Path(self._outdata_uuid(component, input_data) +
                    '.pickle').exists():
            return None, False, ended

        # Successful.
        output_data = pickle.load(
            self._outdata_uuid(component, input_data) + '.pickle'
        )
        return output_data, True, ended

    def store_result_impl(self, component, input_data, output_data):
        pickle.dump(output_data, output_data.uuid + '.pickle')

        # Store dump if requested.
        if self._dump:
            dump_uuid = uuid((component.uuid + component.train_data_uuid__mutable()).encode()





                             )
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
                    config=? and op=? and 
                    dtr=? and din=?
                '''
        set_args = [log_uuid, output_data and output_data.uuid(),
                    component.time_spent, dump_uuid, fail, component.mark]
        where_args = [component.uuid, component.op,
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
        :param history: nested tuples
        :param just_check_exists:
        :return:
        """
        hist_uuid = uuid(zlibext_pack(history))

        sql = f'''
                select 
                    X,Y,Z,P,U,V,W,Q,E,F,l,m,k,C,cols,des
                from 
                    data 
                        left join dataset on dataset=dsid 
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
        for field in flst:
            mid = row[field]
            if mid is not None:
                self.query(f'select val,w,h from mat where mid=?', [mid])
                rone = self.get_one()
                dic[field] = unpack_data(rone['val'], rone['w'], rone['h'])
        return Data(columns=zlibext_unpack(row['cols']), **dic)

    # def get_component_by_uuid(self, component_uuid, just_check_exists=False):
    #     field = 'cfg'
    #     if just_check_exists:
    #         field = '1'
    #     self.query(f'select {field} from config where cid=?', [component_uuid])
    #     result = self.get_one()
    #     if result is None:
    #         return None
    #     if just_check_exists:
    #         return True
    #     return result[field]

    # def get_component(self, component, train_data, input_data,
    #                   just_check_exists=False):
    #     field = 'bytes,cfg'
    #     if just_check_exists:
    #         field = '1'
    #     self.query(f'select {field} '
    #                f'from res'
    #                f'   left join dump on dumpc=duic '
    #                f'   left join config on config=cid '
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
    #                                          **json_unpack(result['cfg']))

    @profile
    def get_data_by_uuid_impl(self, datauuid):
        sql = f'''
                select 
                    X,Y,Z,P,U,V,W,Q,E,F,l,m,k,C,cols,nested,des
                from 
                    data 
                        left join dataset on dataset=dsid 
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
               'history': zlibext_unpack(row['nested'])}
        fields = [Data.from_alias[k]
                  for k, v in row.items() if len(k) == 1 and v is not None]
        for field in fields:
            mid = row[field]
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
                    des, nested
                from
                    res join data on dtr=did 
                        join dataset on dataset=dsid 
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
                row['history'] = zlibext_unpack(row.pop('nested'))
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
