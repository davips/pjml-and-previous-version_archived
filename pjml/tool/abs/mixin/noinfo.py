from typing import Dict, Any


class withNoInfo:
    def _info_impl(self, *args) -> Dict[str, Any]:
        return {}
