"""
Data models for Pharma MMM Agent.

Re-exports Pydantic models from services for convenience.
"""

from ..services.data_service import (
    DataIngestionRequest,
    DataValidationResult,
    DataSummary,
    PreprocessedData,
    AdstockTransformRequest,
    AdstockTransformResult
)

from ..services.modeling_service import (
    ModelTrainingRequest,
    ChannelContribution,
    ModelResults,
    DecompositionResult,
    PredictionRequest,
    PredictionResult
)

from ..services.optimization_service import (
    ChannelConstraint,
    OptimizationRequest,
    OptimizedAllocation,
    OptimizationResult,
    ScenarioRequest,
    ScenarioResult,
    MarginalROIRequest,
    MarginalROIResult
)

from ..services.insight_service import (
    InsightRequest,
    InsightResponse,
    QARequest,
    QAResponse,
    ReportRequest,
    ReportResponse
)

__all__ = [
    # Data models
    "DataIngestionRequest",
    "DataValidationResult",
    "DataSummary",
    "PreprocessedData",
    "AdstockTransformRequest",
    "AdstockTransformResult",
    # Modeling models
    "ModelTrainingRequest",
    "ChannelContribution",
    "ModelResults",
    "DecompositionResult",
    "PredictionRequest",
    "PredictionResult",
    # Optimization models
    "ChannelConstraint",
    "OptimizationRequest",
    "OptimizedAllocation",
    "OptimizationResult",
    "ScenarioRequest",
    "ScenarioResult",
    "MarginalROIRequest",
    "MarginalROIResult",
    # Insight models
    "InsightRequest",
    "InsightResponse",
    "QARequest",
    "QAResponse",
    "ReportRequest",
    "ReportResponse"
]
