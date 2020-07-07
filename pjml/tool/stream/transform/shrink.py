#
#
# class Shrink(ConfigLess):
#     def _apply_impl(self, collection):
#         return self._use_impl(collection)
#
#     def _use_impl(self, collection, **kwargs):
#         newcoll = collection.updated(
#             transformations=self.transformations(step='u'),
#             datas=[d for d in collection if d is not None],
#             failure=collection.failure
#         )
#         if newcoll.size == 0:
#             print('WW: All Nones')
#             return ???
#         return newcoll
