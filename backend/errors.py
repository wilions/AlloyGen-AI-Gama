from __future__ import annotations


class PipelineError(Exception):
    """Base error for pipeline operations."""
    pass


class DataValidationError(PipelineError):
    """Raised when data fails pre-training validation."""
    pass


class ModelTrainingError(PipelineError):
    """Raised when all models fail to train."""
    pass


class TargetNotFoundError(PipelineError):
    """Raised when a target column is not found in the dataset."""
    pass


class PredictionError(PipelineError):
    """Raised when prediction fails."""
    pass
