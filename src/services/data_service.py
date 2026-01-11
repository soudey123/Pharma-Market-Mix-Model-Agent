"""
Data Service for Pharma MMM Agent.

Handles data ingestion, validation, and preprocessing.
Designed as a stateless service with clear input/output contracts
for easy conversion to n8n workflow nodes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from loguru import logger
import json


# ============================================================================
# Input/Output Contracts (Pydantic Models)
# ============================================================================

class DataIngestionRequest(BaseModel):
    """Request model for data ingestion."""
    file_path: Optional[str] = None
    data_json: Optional[str] = None  # JSON string for n8n compatibility
    date_column: str = "date"
    sales_column: str = "sales"
    spend_columns: Optional[List[str]] = None  # Auto-detect if None
    control_columns: Optional[List[str]] = None


class DataValidationResult(BaseModel):
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    date_range: Optional[Dict[str, str]] = None
    missing_values: Dict[str, int] = Field(default_factory=dict)


class DataSummary(BaseModel):
    """Summary statistics for the dataset."""
    row_count: int
    date_range: Dict[str, str]
    total_sales: float
    avg_weekly_sales: float
    channels: List[str]
    channel_spend_summary: Dict[str, Dict[str, float]]
    control_variables: List[str]


class PreprocessedData(BaseModel):
    """Preprocessed data ready for modeling."""
    data_json: str  # JSON string for serialization
    date_column: str
    sales_column: str
    spend_columns: List[str]
    control_columns: List[str]
    n_observations: int

    class Config:
        arbitrary_types_allowed = True


class AdstockTransformRequest(BaseModel):
    """Request for adstock transformation."""
    data_json: str
    spend_column: str
    decay_rate: float = Field(ge=0.0, le=1.0)
    max_lag: int = Field(default=12, ge=1, le=52)
    adstock_type: str = "geometric"  # geometric or weibull

    @field_validator('adstock_type')
    @classmethod
    def validate_adstock_type(cls, v):
        if v not in ["geometric", "weibull"]:
            raise ValueError("adstock_type must be 'geometric' or 'weibull'")
        return v


class AdstockTransformResult(BaseModel):
    """Result of adstock transformation."""
    transformed_column: str
    original_column: str
    decay_rate: float
    adstock_type: str
    data_json: str


# ============================================================================
# Data Service Implementation
# ============================================================================

class DataService:
    """
    Stateless data service for ingestion, validation, and preprocessing.

    All methods are designed with clear input/output contracts for
    easy conversion to FastAPI endpoints and n8n nodes.
    """

    @staticmethod
    def ingest_data(request: DataIngestionRequest) -> Tuple[pd.DataFrame, DataValidationResult]:
        """
        Ingest data from file or JSON string.

        Args:
            request: DataIngestionRequest with file path or JSON data

        Returns:
            Tuple of (DataFrame, DataValidationResult)
        """
        logger.info("Starting data ingestion...")

        try:
            if request.file_path:
                if request.file_path.endswith('.csv'):
                    df = pd.read_csv(request.file_path)
                elif request.file_path.endswith('.xlsx'):
                    df = pd.read_excel(request.file_path)
                elif request.file_path.endswith('.json'):
                    df = pd.read_json(request.file_path)
                else:
                    return pd.DataFrame(), DataValidationResult(
                        is_valid=False,
                        errors=["Unsupported file format. Use CSV, XLSX, or JSON."]
                    )
            elif request.data_json:
                df = pd.read_json(request.data_json)
            else:
                return pd.DataFrame(), DataValidationResult(
                    is_valid=False,
                    errors=["No data source provided. Provide file_path or data_json."]
                )

            # Parse date column
            if request.date_column in df.columns:
                df[request.date_column] = pd.to_datetime(df[request.date_column])

            # Validate the data
            validation = DataService.validate_data(
                df,
                date_column=request.date_column,
                sales_column=request.sales_column,
                spend_columns=request.spend_columns
            )

            logger.info(f"Ingested {len(df)} rows with validation status: {validation.is_valid}")
            return df, validation

        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            return pd.DataFrame(), DataValidationResult(
                is_valid=False,
                errors=[f"Data ingestion failed: {str(e)}"]
            )

    @staticmethod
    def validate_data(
        df: pd.DataFrame,
        date_column: str = "date",
        sales_column: str = "sales",
        spend_columns: Optional[List[str]] = None
    ) -> DataValidationResult:
        """
        Validate dataset for MMM requirements.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            sales_column: Name of sales/response column
            spend_columns: List of spend columns to validate

        Returns:
            DataValidationResult with validation status and details
        """
        errors = []
        warnings = []

        # Check if DataFrame is empty
        if df.empty:
            return DataValidationResult(
                is_valid=False,
                errors=["DataFrame is empty"]
            )

        # Check required columns
        if date_column not in df.columns:
            errors.append(f"Date column '{date_column}' not found")

        if sales_column not in df.columns:
            errors.append(f"Sales column '{sales_column}' not found")

        # Auto-detect spend columns if not provided
        if spend_columns is None:
            spend_columns = [col for col in df.columns if col.endswith('_spend')]
            if not spend_columns:
                warnings.append("No spend columns detected (columns ending with '_spend')")

        # Validate spend columns exist
        missing_spend = [col for col in (spend_columns or []) if col not in df.columns]
        if missing_spend:
            errors.append(f"Spend columns not found: {missing_spend}")

        # Check for minimum data points
        if len(df) < 52:
            warnings.append(f"Dataset has only {len(df)} observations. Recommend at least 52 weeks.")

        # Check for missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: v for k, v in missing_values.items() if v > 0}

        if date_column in missing_values:
            errors.append(f"Date column has {missing_values[date_column]} missing values")

        if sales_column in missing_values:
            warnings.append(f"Sales column has {missing_values[sales_column]} missing values")

        # Check for negative values in sales
        if sales_column in df.columns and (df[sales_column] < 0).any():
            errors.append("Sales column contains negative values")

        # Check date column is datetime
        if date_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                try:
                    pd.to_datetime(df[date_column])
                except:
                    errors.append(f"Cannot parse '{date_column}' as datetime")

        # Get date range
        date_range = None
        if date_column in df.columns and not df[date_column].isnull().all():
            try:
                dates = pd.to_datetime(df[date_column])
                date_range = {
                    "start": dates.min().isoformat(),
                    "end": dates.max().isoformat()
                }
            except:
                pass

        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            column_count=len(df.columns),
            date_range=date_range,
            missing_values=missing_values
        )

    @staticmethod
    def get_data_summary(
        df: pd.DataFrame,
        date_column: str = "date",
        sales_column: str = "sales"
    ) -> DataSummary:
        """
        Generate summary statistics for the dataset.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            sales_column: Name of sales column

        Returns:
            DataSummary with key statistics
        """
        # Identify spend and control columns
        spend_columns = [col for col in df.columns if col.endswith('_spend') and col != 'competitor_spend']
        control_columns = [col for col in df.columns if col in ['competitor_spend', 'price', 'seasonality']]

        # Channel names
        channels = [col.replace('_spend', '').replace('_', ' ').title() for col in spend_columns]

        # Spend summary per channel
        channel_spend_summary = {}
        for col in spend_columns:
            channel_name = col.replace('_spend', '').replace('_', ' ').title()
            channel_spend_summary[channel_name] = {
                "total": float(df[col].sum()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }

        # Date range
        dates = pd.to_datetime(df[date_column])
        date_range = {
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d")
        }

        return DataSummary(
            row_count=len(df),
            date_range=date_range,
            total_sales=float(df[sales_column].sum()),
            avg_weekly_sales=float(df[sales_column].mean()),
            channels=channels,
            channel_spend_summary=channel_spend_summary,
            control_variables=control_columns
        )

    @staticmethod
    def preprocess_data(
        df: pd.DataFrame,
        date_column: str = "date",
        sales_column: str = "sales",
        spend_columns: Optional[List[str]] = None,
        control_columns: Optional[List[str]] = None,
        normalize: bool = True
    ) -> PreprocessedData:
        """
        Preprocess data for modeling.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            sales_column: Name of sales column
            spend_columns: List of spend columns
            control_columns: List of control variable columns
            normalize: Whether to normalize features

        Returns:
            PreprocessedData object with processed data
        """
        df = df.copy()

        # Auto-detect columns if not provided
        if spend_columns is None:
            spend_columns = [col for col in df.columns if col.endswith('_spend') and col != 'competitor_spend']

        if control_columns is None:
            control_columns = [col for col in df.columns if col in ['competitor_spend', 'price']]

        # Handle missing values
        for col in spend_columns + control_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        if sales_column in df.columns:
            df[sales_column] = df[sales_column].fillna(df[sales_column].median())

        # Sort by date
        df = df.sort_values(date_column).reset_index(drop=True)

        # Normalize if requested
        if normalize:
            for col in spend_columns + control_columns:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        df[f"{col}_normalized"] = (df[col] - mean_val) / std_val

        # Convert to JSON for serialization
        data_json = df.to_json(orient='records', date_format='iso')

        return PreprocessedData(
            data_json=data_json,
            date_column=date_column,
            sales_column=sales_column,
            spend_columns=spend_columns,
            control_columns=control_columns,
            n_observations=len(df)
        )

    @staticmethod
    def apply_adstock_transform(
        request: AdstockTransformRequest
    ) -> AdstockTransformResult:
        """
        Apply adstock transformation to a spend column.

        Args:
            request: AdstockTransformRequest with transformation parameters

        Returns:
            AdstockTransformResult with transformed data
        """
        df = pd.read_json(request.data_json)
        spend = df[request.spend_column].values

        if request.adstock_type == "geometric":
            transformed = DataService._geometric_adstock(
                spend,
                request.decay_rate,
                request.max_lag
            )
        else:  # weibull
            transformed = DataService._weibull_adstock(
                spend,
                request.decay_rate,
                request.max_lag
            )

        new_col = f"{request.spend_column}_adstock"
        df[new_col] = transformed

        return AdstockTransformResult(
            transformed_column=new_col,
            original_column=request.spend_column,
            decay_rate=request.decay_rate,
            adstock_type=request.adstock_type,
            data_json=df.to_json(orient='records', date_format='iso')
        )

    @staticmethod
    def _geometric_adstock(
        x: np.ndarray,
        decay: float,
        max_lag: int
    ) -> np.ndarray:
        """Apply geometric adstock transformation."""
        adstocked = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            for j in range(min(i + 1, max_lag)):
                adstocked[i] += x[i - j] * (decay ** j)
        return adstocked

    @staticmethod
    def _weibull_adstock(
        x: np.ndarray,
        decay: float,
        max_lag: int,
        shape: float = 2.0
    ) -> np.ndarray:
        """
        Apply Weibull adstock transformation.

        Weibull allows for delayed peak effects, useful for pharma
        where effect may build over time.
        """
        from scipy.stats import weibull_min

        weights = np.array([
            weibull_min.pdf(j, shape, scale=1/decay)
            for j in range(max_lag)
        ])
        weights = weights / weights.sum()  # Normalize

        adstocked = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            for j in range(min(i + 1, max_lag)):
                adstocked[i] += x[i - j] * weights[j]

        return adstocked

    @staticmethod
    def apply_saturation(
        data_json: str,
        column: str,
        k: float = 0.5,
        s: float = 1.0
    ) -> str:
        """
        Apply Hill saturation function.

        Args:
            data_json: JSON string of DataFrame
            column: Column to transform
            k: Half-saturation point
            s: Shape parameter

        Returns:
            JSON string with new saturated column
        """
        df = pd.read_json(data_json)
        x = df[column].values

        # Normalize to 0-1
        x_norm = x / (x.max() + 1e-8)

        # Hill function
        saturated = x_norm ** s / (k ** s + x_norm ** s)

        new_col = f"{column}_saturated"
        df[new_col] = saturated

        return df.to_json(orient='records', date_format='iso')
