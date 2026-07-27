"""
MOKO Context Pager (Marathon adapter)
======================================
Wrapper marathon di atas TokenStreamPager — API legacy tetap kompatibel.
"""
from moko_marathon.token_stream_pager import TokenStreamPager


class ContextPager(TokenStreamPager):
    """Pager khusus marathon — delegasi ke TokenStreamPager profile marathon."""

    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, stream_kind="marathon")

    def build_active_context(self, goal: str, retrieved_context: str = "") -> str:
        return super().build_active_context(
            goal, retrieved_context, profile="marathon"
        )
