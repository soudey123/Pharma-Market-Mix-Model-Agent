"""
Data Agent for Pharma MMM Agent.

Coordinates data ingestion, validation, and preprocessing tasks.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
import pandas as pd
from loguru import logger

from ..services.data_service import (
    DataService, DataIngestionRequest, DataValidationResult,
    DataSummary, PreprocessedData
)


class DataAgentRequest(BaseModel):
    """Request for Data Agent operations."""
    operation: str  # "ingest", "validate", "preprocess", "full_pipeline"
    file_path: Optional[str] = None
    data_json: Optional[str] = None
    date_column: str = "date"
    sales_column: str = "sales"
    spend_columns: Optional[List[str]] = None
    control_columns: Optional[List[str]] = None


class DataAgentResponse(BaseModel):
    """Response from Data Agent operations."""
    success: bool
    operation: str
    message: str
    validation: Optional[DataValidationResult] = None
    summary: Optional[DataSummary] = None
    preprocessed_data: Optional[PreprocessedData] = None
    data_json: Optional[str] = None


class DataAgent:
    """
    Agent for handling data operations.

    Coordinates the DataService to perform complex data tasks
    with error handling and logging.
    """

    def __init__(self):
        self.service = DataService()

    def execute(self, request: DataAgentRequest) -> DataAgentResponse:
        """Execute a data operation."""
        logger.info(f"DataAgent executing operation: {request.operation}")

        if request.operation == "ingest":
            return self._ingest(request)
        elif request.operation == "validate":
            return self._validate(request)
        elif request.operation == "preprocess":
            return self._preprocess(request)
        elif request.operation == "full_pipeline":
            return self._full_pipeline(request)
        else:
            return DataAgentResponse(
                success=False,
                operation=request.operation,
                message=f"Unknown operation: {request.operation}"
            )

    def _ingest(self, request: DataAgentRequest) -> DataAgentResponse:
        """Ingest and validate data."""
        try:
            ingestion_request = DataIngestionRequest(
                file_path=request.file_path,
                data_json=request.data_json,
                date_column=request.date_column,
                sales_column=request.sales_column,
                spend_columns=request.spend_columns
            )

            df, validation = DataService.ingest_data(ingestion_request)

            return DataAgentResponse(
                success=validation.is_valid,
                operation="ingest",
                message="Data ingested successfully" if validation.is_valid else "Validation failed",
                validation=validation,
                data_json=df.to_json(orient='records', date_format='iso') if not df.empty else None
            )

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return DataAgentResponse(
                success=False,
                operation="ingest",
                message=f"Ingestion failed: {str(e)}"
            )

    def _validate(self, request: DataAgentRequest) -> DataAgentResponse:
        """Validate data only."""
        try:
            if request.data_json:
                df = pd.read_json(request.data_json)
            else:
                return DataAgentResponse(
                    success=False,
                    operation="validate",
                    message="No data provided for validation"
                )

            validation = DataService.validate_data(
                df,
                request.date_column,
                request.sales_column,
                request.spend_columns
            )

            return DataAgentResponse(
                success=validation.is_valid,
                operation="validate",
                message="Validation complete",
                validation=validation
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return DataAgentResponse(
                success=False,
                operation="validate",
                message=f"Validation failed: {str(e)}"
            )

    def _preprocess(self, request: DataAgentRequest) -> DataAgentResponse:
        """Preprocess data for modeling."""
        try:
            if request.data_json:
                df = pd.read_json(request.data_json)
            else:
                return DataAgentResponse(
                    success=False,
                    operation="preprocess",
                    message="No data provided for preprocessing"
                )

            preprocessed = DataService.preprocess_data(
                df,
                request.date_column,
                request.sales_column,
                request.spend_columns,
                request.control_columns
            )

            return DataAgentResponse(
                success=True,
                operation="preprocess",
                message="Preprocessing complete",
                preprocessed_data=preprocessed
            )

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return DataAgentResponse(
                success=False,
                operation="preprocess",
                message=f"Preprocessing failed: {str(e)}"
            )

    def _full_pipeline(self, request: DataAgentRequest) -> DataAgentResponse:
        """Run full data pipeline: ingest, validate, summarize, preprocess."""
        try:
            # Step 1: Ingest
            ingestion_request = DataIngestionRequest(
                file_path=request.file_path,
                data_json=request.data_json,
                date_column=request.date_column,
                sales_column=request.sales_column,
                spend_columns=request.spend_columns
            )

            df, validation = DataService.ingest_data(ingestion_request)

            if not validation.is_valid:
                return DataAgentResponse(
                    success=False,
                    operation="full_pipeline",
                    message="Validation failed during pipeline",
                    validation=validation
                )

            # Step 2: Get summary
            summary = DataService.get_data_summary(
                df,
                request.date_column,
                request.sales_column
            )

            # Step 3: Preprocess
            preprocessed = DataService.preprocess_data(
                df,
                request.date_column,
                request.sales_column,
                request.spend_columns,
                request.control_columns
            )

            return DataAgentResponse(
                success=True,
                operation="full_pipeline",
                message="Full pipeline complete",
                validation=validation,
                summary=summary,
                preprocessed_data=preprocessed,
                data_json=df.to_json(orient='records', date_format='iso')
            )

        except Exception as e:
            logger.error(f"Full pipeline failed: {e}")
            return DataAgentResponse(
                success=False,
                operation="full_pipeline",
                message=f"Pipeline failed: {str(e)}"
            )
