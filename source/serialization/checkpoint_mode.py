from enum import Enum


class CheckpointMode(Enum):
    LATEST = "latest"
    CUSTOM = "custom"
    NONE = "none"
