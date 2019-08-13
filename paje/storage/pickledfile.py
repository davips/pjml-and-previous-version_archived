import _pickle as pickle
from pathlib import Path

from paje.storage.cache import Cache


class PickledFile(Cache):
    @staticmethod
    def _resfile(component, input_data):
        return f'{component.uuid}-{component.train_data_uuid__mutable()}' \
               f'-{component.op}-{input_data.name_uuid()}'

    def store_data_impl(self, data, file=None):
        pickle.dump(data, open((file or data.uuid()) + '.dump', 'wb'))

    def lock_impl(self, component, input_data):
        print(self.name, 'Locking...\n',
              self._resfile(component, input_data))
        Path(self._resfile(component, input_data) + '.dump').touch()

    def get_result_impl(self, component, input_data):
        # Not available from previous attempts?
        if component.failed or component.locked_by_others:
            print(self.name, 'W: Previously failed or locked.')
            return None, True, component.failed is not None

        # Failed?
        file = self._resfile(component, input_data)
        if Path(file + '.fail').exists():
            print(self.name, 'W: Failed.', file)
            component.failed = True
            ended = component.failed is not None
            return None, True, ended

        # Not started yet?
        if not Path(file + '.dump').exists():
            print(self.name, 'W: Not started.', file)
            return None, False, False

        # Locked?
        if Path(file + '.dump').stat().st_size == 0:
            print(self.name, 'W: Locked.', file)
            component.locked_by_others = True
            return None, True, False

        # Successful.
        print(self.name, 'Successful.', file)
        output_data = pickle.load(open(file + '.dump', 'rb'))
        component.failed = False
        component.time_spent = -1
        component.host = 'unknown'
        return output_data, True, True

    def store_result_impl(self, component, input_data, output_data):
        print(self.name, 'NOT dumping inputdata..')
        # pickle.dump(input_data, open(input_data.uuid + '.dump', 'wb'))

        # Store dump if requested.
        if self._dump:
            pickle.dump(
                component, open(self._resfile(component, input_data) +
                                '-comp.dump', 'wb')
            )

        # Store a log if apply() failed.
        if component.failed:
            print('failing........................')
            fw = open(self._resfile(component, input_data) + '.log', 'w')
            fw.write(component.failure)
            fw.close()
            Path(self._resfile(component, input_data) + '.fail').touch()

        # Unlock and save result.
        self.store_data_impl(output_data, self._resfile(component, input_data))

        print(self.name, 'Stored!\n', self._resfile(component, input_data))

    def get_data_by_name_impl(self, name, fields=None, history=None):
        raise NotImplementedError('todo')

    #     """
    #     To just recover the original dataset you can pass history=None.
    #     Specify fields if you want to reduce traffic, otherwise all available
    #     fields will be fetched.
    #
    #     ps. 1: Obviously, when getting prediction data (i.e., results),
    #      the history which led to the predictions should be provided.
    #     :param name:
    #     :param fields: None=get full Data; case insensitive; e.g. 'X,y,Z'
    #     :param history: nested tuples
    #     :param just_check_exists:
    #     :return:
    #     """
    #     hist_uuid = uuid(zlibext_pack(history))
    #
    #     sql = f'''
    #             select
    #                 X,Y,Z,P,U,V,W,Q,E,F,l,m,k,C,cols,des
    #             from
    #                 data
    #                     left join dataset on dataset=dsid
    #                     left join attr on attr=aid
    #             where
    #                 des=? and hist=?'''
    #     self.query(sql, [name, hist_uuid])
    #     row = self.get_one()
    #     if row is None:
    #         return None
    #
    #     # Recover requested matrices/vectors.
    #     dic = {'name': name, 'history': history}
    #     if fields is None:
    #         flst = [k for k, v in row.items() if len(k) == 1 and v is not None]
    #     else:
    #         flst = fields.split(',')
    #     for field in flst:
    #         mid = row[field]
    #         if mid is not None:
    #             self.query(f'select val,w,h from mat where mid=?', [mid])
    #             rone = self.get_one()
    #             dic[field] = unpack_data(rone['val'], rone['w'], rone['h'])
    #     return Data(columns=zlibext_unpack(row['cols']), **dic)

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

    def get_data_by_uuid_impl(self, datauuid):
        raise NotImplementedError('Storage in dump mode (pickledfile) cannot'
                                  ' recover training data from storage')

    #     sql = f'''
    #             select
    #                 X,Y,Z,P,U,V,W,Q,E,F,l,m,k,C,cols,nested,des
    #             from
    #                 data
    #                     left join dataset on dataset=dsid
    #                     left join hist on hist=hid
    #                     left join attr on attr=aid
    #             where
    #                 did=?'''
    #     self.query(sql, [datauuid])
    #     row = self.get_one()
    #     if row is None:
    #         return None
    #
    #     # Recover requested matrices/vectors.
    #     # TODO: surely there is duplicated code to be refactored in this file!
    #     dic = {'name': row['des'],
    #            'history': zlibext_unpack(row['nested'])}
    #     fields = [Data.from_alias[k]
    #               for k, v in row.items() if len(k) == 1 and v is not None]
    #     for field in fields:
    #         mid = row[field]
    #         if mid is not None:
    #             self.query(f'select val,w,h from mat where mid=?', [mid])
    #             rone = self.get_one()
    #             dic[field] = unpack_data(rone['val'], rone['w'], rone['h'])
    #     return Data(columns=zlibext_unpack(row['cols']), **dic)
    #
    def get_finished_names_by_mark_impl(self, mark):
        raise NotImplementedError('todo')

    #     """
    #     Finished means nonfailed and unlocked results.
    #     The original datasets will not be returned.
    #     original dataset = stored with no history of transformations.
    #     All results are returned (apply & use, as many as there are),
    #      so the names come along with their history,
    #     :param mark:
    #     :return: [dicts]
    #     """
    #     self.query(f"""
    #             select
    #                 des, nested
    #             from
    #                 res join data on dtr=did
    #                     join dataset on dataset=dsid
    #                     join hist on hist=hid
    #             where
    #                 end!='0000-00-00 00:00:00' and
    #                 fail=0 and mark=?
    #         """, [mark])
    #     rows = self.get_all()
    #     if rows is None:
    #         return None
    #     else:
    #         for row in rows:
    #             row['name'] = row.pop('des')
    #             row['history'] = zlibext_unpack(row.pop('nested'))
    #         return rows
