
class CameraError(Exception):
    """Base class for camera-related issues."""


class CameraOpenError(Exception):
    """Raised when the OpenCV video capture device fails to open."""

    def __init__(self, message, device_index=None, error_code=None):
        # 1. Pass the primary message to the base Exception class
        super().__init__(message)

        # 2. Attach custom attributes for debugging and logging
        self.message = message
        self.device_index = device_index
        self.error_code = error_code

    def __str__(self):
        # 3. Customize how the error prints in logs/terminal
        base_msg = f"[Error {self.error_code}] " if self.error_code else ""
        device_msg = f" (Device Index: {self.device_index})" if self.device_index is not None else ""
        return f"{base_msg}{self.message}{device_msg}"

class CameraFrameReadError(CameraError):
    """Raised when a frame cannot be read from an active stream."""
    pass

class ProcessingError(Exception):
    """Raised when image processing steps fail."""
    pass