"""
Modeling Agent for Pharma MMM Agent.

Coordinates model training, prediction, and decomposition tasks.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel
from loguru import logger

from ..services.modeling_service import (
    ModelingService, ModelTrainingRequest, ModelResults,
    PredictionRequest, PredictionResult, DecompositionResult
)


class ModelingAgentRequest(BaseModel):
    """Request for Modeling Agent operations."""
    operation: str  # "train", "predict", "decompose"
    # Training params
    data_json: Optional[str] = None
    date_column: str = "date"
    sales_column: str = "sales"
    spend_columns: Optional[List[str]] = None
    control_columns: Optional[List[str]] = None
    adstock_max_lag: int = 12
    estimate_adstock: bool = True
    apply_saturation: bool = True
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 2
    # Prediction/decomposition params
    model_id: Optional[str] = None
    spend_scenario_json: Optional[str] = None


class ModelingAgentResponse(BaseModel):
    """Response from Modeling Agent operations."""
    success: bool
    operation: str
    message: str
    model_results: Optional[ModelResults] = None
    prediction_results: Optional[PredictionResult] = None
    decomposition: Optional[List[Dict]] = None


class ModelingAgent:
    """
    Agent for handling modeling operations.

    Coordinates the ModelingService to train models,
    make predictions, and generate decompositions.
    """

    def __init__(self):
        self.service = ModelingService()

    def execute(self, request: ModelingAgentRequest) -> ModelingAgentResponse:
        """Execute a modeling operation."""
        logger.info(f"ModelingAgent executing operation: {request.operation}")

        if request.operation == "train":
            return self._train(request)
        elif request.operation == "predict":
            return self._predict(request)
        elif request.operation == "decompose":
            return self._decompose(request)
        else:
            return ModelingAgentResponse(
                success=False,
                operation=request.operation,
                message=f"Unknown operation: {request.operation}"
            )

    def _train(self, request: ModelingAgentRequest) -> ModelingAgentResponse:
        """Train a new model."""
        try:
            if not request.data_json:
                return ModelingAgentResponse(
                    success=False,
                    operation="train",
                    message="No data provided for training"
                )

            if not request.spend_columns:
                return ModelingAgentResponse(
                    success=False,
                    operation="train",
                    message="No spend columns specified"
                )

            training_request = ModelTrainingRequest(
                data_json=request.data_json,
                date_column=request.date_column,
                sales_column=request.sales_column,
                spend_columns=request.spend_columns,
                control_columns=request.control_columns,
                adstock_max_lag=request.adstock_max_lag,
                estimate_adstock=request.estimate_adstock,
                apply_saturation=request.apply_saturation,
                n_samples=request.n_samples,
                n_tune=request.n_tune,
                n_chains=request.n_chains
            )

            logger.info("Starting model training...")
            results = ModelingService.train_model(training_request)

            return ModelingAgentResponse(
                success=results.success,
                operation="train",
                message=f"Model trained successfully. ID: {results.model_id}" if results.success else "Training failed",
                model_results=results
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return ModelingAgentResponse(
                success=False,
                operation="train",
                message=f"Training failed: {str(e)}"
            )

    def _predict(self, request: ModelingAgentRequest) -> ModelingAgentResponse:
        """Make predictions with a trained model."""
        try:
            if not request.model_id:
                return ModelingAgentResponse(
                    success=False,
                    operation="predict",
                    message="No model_id provided"
                )

            if not request.spend_scenario_json:
                return ModelingAgentResponse(
                    success=False,
                    operation="predict",
                    message="No spend scenario provided"
                )

            pred_request = PredictionRequest(
                model_id=request.model_id,
                spend_scenario_json=request.spend_scenario_json
            )

            results = ModelingService.predict(pred_request)

            return ModelingAgentResponse(
                success=True,
                operation="predict",
                message="Predictions generated successfully",
                prediction_results=results
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ModelingAgentResponse(
                success=False,
                operation="predict",
                message=f"Prediction failed: {str(e)}"
            )

    def _decompose(self, request: ModelingAgentRequest) -> ModelingAgentResponse:
        """Get sales decomposition for a model."""
        try:
            if not request.model_id:
                return ModelingAgentResponse(
                    success=False,
                    operation="decompose",
                    message="No model_id provided"
                )

            decomposition = ModelingService.get_decomposition(request.model_id)

            return ModelingAgentResponse(
                success=True,
                operation="decompose",
                message="Decomposition generated successfully",
                decomposition=[d.model_dump() for d in decomposition]
            )

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return ModelingAgentResponse(
                success=False,
                operation="decompose",
                message=f"Decomposition failed: {str(e)}"
            )
