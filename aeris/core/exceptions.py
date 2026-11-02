"""
aeris.core.exceptions
=====================
Structured exception hierarchy for AERIS.

All exceptions inherit from AerisError, allowing callers to catch
all AERIS-specific errors with a single except clause.
"""

class AerisError(Exception):
    """Base exception for all AERIS errors."""
    pass


class ValidationError(AerisError):
    """Raised when design variable validation fails."""
    
    def __init__(self, field: str, message: str, value=None):
        self.field = field
        self.value = value
        super().__init__(f"Validation error in '{field}': {message}")


class ModelValidityError(AerisError):
    """Raised when operating conditions are outside model validity."""
    
    def __init__(self, check_name: str, status: str, message: str, 
                 threshold=None, actual=None):
        self.check_name = check_name
        self.status = status  # "WARNING" or "INVALID"
        self.threshold = threshold
        self.actual = actual
        super().__init__(
            f"Model validity {status}: {check_name} - {message}"
        )


class GeometryError(AerisError):
    """Raised when geometry generation or validation fails."""
    pass


class OptimizationError(AerisError):
    """Raised when optimization fails or converges unexpectedly."""
    pass


class ConfigurationError(AerisError):
    """Raised when configuration is invalid or missing."""
    pass


class CADExportError(AerisError):
    """Raised when CAD export fails."""
    pass