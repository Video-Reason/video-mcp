# Importing this package triggers dataset adapter registration so that
# ``list_adapters()`` / ``get_adapter()`` always see every built-in adapter.
import video_mcp.datasets  # noqa: F401

__all__ = [
    "adapter",
    "build_video_mcp",
    "build_video_mcp_clips",
    "video_mcp_format",
]
