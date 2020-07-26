# from pjdata.mixin.serialization import withSerialization
# from pjdata.transformer.enhancer import Enhancer
# from pjdata.transformer.pholder import PHolder
# from pjdata.transformer.transformer import Transformer
#
#
# class withDefaultEnhancerImpl(withSerialization):  # withSerialization need because of Enhancer(*self* ... )
#     def _enhancer_impl(self) -> Transformer:
#         return PHolder(self)
#
#     def _cfuuid_impl(self):
#         raise Exception(
#             "This method should be overriden by a child class. HINT: put the mixin in the last inheritance position"
#         )
#
#     def _name_impl(self):
#         raise Exception(
#             "This method should be overriden by a child class. HINT: put the mixin in the last inheritance position"
#         )
#
#     def _uuid_impl(self):
#         raise Exception(
#             "This method should be overriden by a child class. HINT: put the mixin in the last inheritance position"
#         )
