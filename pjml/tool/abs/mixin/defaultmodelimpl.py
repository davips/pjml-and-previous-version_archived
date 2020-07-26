# from abc import abstractmethod
# from functools import lru_cache
# from typing import Any, Dict, Callable
#
# from pjdata.mixin.serialization import withSerialization
# from pjdata.transformer.pholder import PHolder
# from pjdata.transformer.transformer import Transformer
# from pjdata.types import Data
#
#
# class withDefaultModelImpl(withSerialization):  # withSerialization need because of Enhancer(*self* ... )
#     def _modelsvsvsvs_impl(self) -> Transformer:
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
