"""Custom exception classes for the atmospheric visibility enhancement system.

Ensures explicit and descriptive error reporting rather than silent failures
or unhandled None returns from lower-level libraries like OpenCV.
"""


class DehazingProjectError(Exception):
    """Base exception for all domain-specific errors in this project."""
    pass


class ImageLoadError(DehazingProjectError):
    """Raised when an image fails to load from the specified path or format."""
    pass


class ImageSaveError(DehazingProjectError):
    """Raised when an image cannot be saved to the destination path."""
    pass


class DeviceError(DehazingProjectError):
    """Raised when an unsupported or unavailable compute device is requested."""
    pass


class PipelineError(DehazingProjectError):
    """Raised when an operation in an image/video processing pipeline fails."""
    pass
