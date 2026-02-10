# Import adapter submodules so their @register_adapter decorators execute.
from video_mcp.datasets import corecognition as _corecognition  # noqa: F401
from video_mcp.datasets import scienceqa as _scienceqa  # noqa: F401

__all__ = ["corecognition", "scienceqa"]
