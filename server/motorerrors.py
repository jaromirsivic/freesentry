class MotorControlError(RuntimeError):
    """Base class for managed motor control failures."""


class MotorOverrideConflictError(MotorControlError):
    """Raised when a manual hardware override conflicts with managed control."""
