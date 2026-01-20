"""Custom exceptions for genome_entropy."""


class OrfEntropyError(Exception):
    """Base exception for genome_entropy package."""

    pass


class ConfigurationError(OrfEntropyError):
    """Raised when there's a configuration error."""

    pass


class InputError(OrfEntropyError):
    """Raised when input data is invalid or cannot be processed."""

    pass


class OrfFinderError(OrfEntropyError):
    """Raised when ORF finding fails."""

    pass


class TranslationError(OrfEntropyError):
    """Raised when translation fails."""

    pass


class EncodingError(OrfEntropyError):
    """Raised when 3Di encoding fails."""

    pass


class ModelError(OrfEntropyError):
    """Raised when model loading or inference fails."""

    pass


class DeviceError(OrfEntropyError):
    """Raised when device selection or initialization fails."""

    pass


class PipelineError(OrfEntropyError):
    """Raised when the pipeline orchestration fails."""

    pass
