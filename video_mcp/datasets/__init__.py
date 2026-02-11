# Import adapter submodules so their @register_adapter decorators execute.
from video_mcp.datasets import corecognition as _corecognition  # noqa: F401
from video_mcp.datasets import mathvision as _mathvision  # noqa: F401
from video_mcp.datasets import phyx as _phyx  # noqa: F401
from video_mcp.datasets import scienceqa as _scienceqa  # noqa: F401

__all__ = ["corecognition", "mathvision", "phyx", "scienceqa"]
