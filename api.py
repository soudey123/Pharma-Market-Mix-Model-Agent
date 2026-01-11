"""
Pharma Market Mix Model Agent - FastAPI Application

REST API endpoints for n8n workflow integration.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.data_service import (
    DataService, DataIngestionRequest, DataValidationResult,
    DataSummary, PreprocessedData, AdstockTransformRequest, AdstockTransformResult
)
from src.services.modeling_service import (
    ModelingService, ModelTrainingRequest, ModelResults,
    PredictionRequest, PredictionResult
)
from src.services.optimization_service import (
    OptimizationService, OptimizationRequest, OptimizationResult,
    ScenarioRequest, ScenarioResult, MarginalROIRequest, MarginalROIResult
)
from src.services.insight_service import (
    InsightService, InsightRequest, InsightResponse,
    QARequest, QAResponse, ReportRequest, ReportResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Pharma MMM Agent API",
    description="""
    REST API for Pharmaceutical Market Mix Modeling.

    This API provides endpoints for:
    - **Data Operations**: Ingestion, validation, preprocessing
    - **Model Training**: Bayesian MMM with PyMC
    - **Optimization**: Budget allocation optimization
    - **Insights**: Claude-powered natural language analysis

    Designed for easy integration with n8n workflows.
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for n8n compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {"status": "healthy", "service": "Pharma MMM Agent API", "version": "0.1.0"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "services": {
            "data_service": "available",
            "modeling_service": "available",
            "optimization_service": "available",
            "insight_service": "available" if os.getenv("ANTHROPIC_API_KEY") else "requires_api_key"
        }
    }


# ============================================================================
# Data Endpoints
# ============================================================================

@app.post("/api/v1/data/validate", response_model=DataValidationResult, tags=["Data"])
async def validate_data(request: DataIngestionRequest):
    """
    Validate uploaded data for MMM requirements.

    This endpoint checks:
    - Required columns exist
    - Data types are correct
    - No missing values in critical columns
    - Sufficient data points

    **n8n Integration**: Use HTTP Request node with POST method.
    """
    try:
        df, validation = DataService.ingest_data(request)
        return validation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class DataSummaryRequest(BaseModel):
    """Request model for data summary."""
    data_json: str
    date_column: str = "date"
    sales_column: str = "sales"


@app.post("/api/v1/data/summary", response_model=DataSummary, tags=["Data"])
async def get_data_summary(request: DataSummaryRequest):
    """
    Get summary statistics for the dataset.

    Returns channel spend summaries, date range, and overall metrics.

    **n8n Integration**: Useful for building dashboards or alerts.
    """
    try:
        import pandas as pd
        df = pd.read_json(request.data_json)
        return DataService.get_data_summary(df, request.date_column, request.sales_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class PreprocessRequest(BaseModel):
    """Request for data preprocessing."""
    data_json: str
    date_column: str = "date"
    sales_column: str = "sales"
    spend_columns: Optional[List[str]] = None
    control_columns: Optional[List[str]] = None
    normalize: bool = True


@app.post("/api/v1/data/preprocess", response_model=PreprocessedData, tags=["Data"])
async def preprocess_data(request: PreprocessRequest):
    """
    Preprocess data for modeling.

    Handles missing values, normalization, and column selection.

    **n8n Integration**: Chain this with model training.
    """
    try:
        import pandas as pd
        df = pd.read_json(request.data_json)
        return DataService.preprocess_data(
            df,
            request.date_column,
            request.sales_column,
            request.spend_columns,
            request.control_columns,
            request.normalize
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/data/adstock", response_model=AdstockTransformResult, tags=["Data"])
async def apply_adstock(request: AdstockTransformRequest):
    """
    Apply adstock transformation to a spend column.

    Supports geometric and Weibull transformations.

    **n8n Integration**: Use for custom feature engineering.
    """
    try:
        return DataService.apply_adstock_transform(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Model Endpoints
# ============================================================================

class TrainingStatus(BaseModel):
    """Training job status."""
    job_id: str
    status: str
    progress: int
    message: str


# Simple job storage (in production, use Redis or database)
training_jobs: Dict[str, Dict] = {}


@app.post("/api/v1/model/train", response_model=ModelResults, tags=["Model"])
async def train_model(request: ModelTrainingRequest):
    """
    Train a Bayesian MMM model.

    This is a synchronous endpoint that blocks until training completes.
    For large datasets, consider using the async training endpoint.

    **n8n Integration**: Use with appropriate timeout settings.
    """
    try:
        results = ModelingService.train_model(request)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/model/train/async", tags=["Model"])
async def train_model_async(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """
    Start async model training.

    Returns a job ID immediately. Use /api/v1/model/status/{job_id} to check progress.

    **n8n Integration**: Use with polling or webhook callback.
    """
    import uuid

    job_id = str(uuid.uuid4())[:8]
    training_jobs[job_id] = {"status": "pending", "progress": 0, "message": "Queued"}

    def train_in_background(job_id: str, request: ModelTrainingRequest):
        try:
            training_jobs[job_id] = {"status": "running", "progress": 10, "message": "Starting"}
            results = ModelingService.train_model(request)
            training_jobs[job_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Training complete",
                "results": results.model_dump()
            }
        except Exception as e:
            training_jobs[job_id] = {"status": "failed", "progress": 0, "message": str(e)}

    background_tasks.add_task(train_in_background, job_id, request)

    return {"job_id": job_id, "status": "accepted"}


@app.get("/api/v1/model/status/{job_id}", tags=["Model"])
async def get_training_status(job_id: str):
    """
    Get training job status.

    **n8n Integration**: Poll this endpoint to check training progress.
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]


@app.post("/api/v1/model/predict", response_model=PredictionResult, tags=["Model"])
async def predict(request: PredictionRequest):
    """
    Make predictions with a trained model.

    Provide spend scenarios to get predicted sales.

    **n8n Integration**: Use for scenario planning workflows.
    """
    try:
        return ModelingService.predict(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/model/{model_id}/decomposition", tags=["Model"])
async def get_decomposition(model_id: str):
    """
    Get sales decomposition for a trained model.

    Returns contribution of each channel over time.

    **n8n Integration**: Use for reporting workflows.
    """
    try:
        decomposition = ModelingService.get_decomposition(model_id)
        return [d.model_dump() for d in decomposition]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Optimization Endpoints
# ============================================================================

@app.post("/api/v1/optimize/budget", response_model=OptimizationResult, tags=["Optimization"])
async def optimize_budget(request: OptimizationRequest):
    """
    Optimize budget allocation across channels.

    Uses trained model response curves to find optimal spend mix.

    **n8n Integration**: Trigger on budget planning cycles.
    """
    try:
        return OptimizationService.optimize_budget(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/optimize/scenarios", response_model=List[ScenarioResult], tags=["Optimization"])
async def analyze_scenarios(request: ScenarioRequest):
    """
    Analyze multiple budget scenarios.

    Compare different allocations to find best strategy.

    **n8n Integration**: Use for what-if analysis workflows.
    """
    try:
        return OptimizationService.analyze_scenarios(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/optimize/marginal-roi", response_model=MarginalROIResult, tags=["Optimization"])
async def get_marginal_roi(request: MarginalROIRequest):
    """
    Get marginal ROI curve for a channel.

    Shows diminishing returns at different spend levels.

    **n8n Integration**: Use for channel-level deep dives.
    """
    try:
        return OptimizationService.get_marginal_roi_curve(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Insight Endpoints
# ============================================================================

insight_service = InsightService()


@app.post("/api/v1/insights/generate", response_model=InsightResponse, tags=["Insights"])
async def generate_insights(request: InsightRequest):
    """
    Generate AI-powered insights from model results.

    Uses Claude to provide natural language analysis.

    **n8n Integration**: Use for automated reporting.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="Anthropic API key not configured")

    try:
        return insight_service.generate_insight(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/insights/qa", response_model=QAResponse, tags=["Insights"])
async def answer_question(request: QARequest):
    """
    Answer questions about model results.

    Natural language Q&A powered by Claude.

    **n8n Integration**: Connect to chatbot or Slack workflows.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="Anthropic API key not configured")

    try:
        return insight_service.answer_question(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/insights/report", response_model=ReportResponse, tags=["Insights"])
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive analysis report.

    Creates full report with multiple sections.

    **n8n Integration**: Use for periodic reporting workflows.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="Anthropic API key not configured")

    try:
        return insight_service.generate_report(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.post("/api/v1/generate-sample-data", tags=["Utilities"])
async def generate_sample_data(n_weeks: int = 156):
    """
    Generate sample pharmaceutical marketing data.

    Creates realistic synthetic data for testing.

    **n8n Integration**: Use for demos or testing workflows.
    """
    try:
        from src.utils.data_generator import PharmaDataGenerator
        import json

        generator = PharmaDataGenerator(n_weeks=n_weeks)
        data, true_effects = generator.generate()

        return {
            "data_json": data.to_json(orient='records', date_format='iso'),
            "n_rows": len(data),
            "columns": data.columns.tolist(),
            "true_effects": true_effects
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/channels", tags=["Utilities"])
async def get_channel_configs():
    """
    Get pharma channel configurations.

    Returns default settings for pharma marketing channels.

    **n8n Integration**: Use to populate form options.
    """
    from src.config import PharmaChannels

    return [
        {
            "name": ch.name,
            "adstock_type": ch.adstock_type,
            "adstock_decay_range": [ch.adstock_decay_min, ch.adstock_decay_max],
            "carryover_weeks": ch.carryover_weeks,
            "typical_spend_range": list(ch.typical_spend_range)
        }
        for ch in PharmaChannels.get_all()
    ]


# ============================================================================
# Webhook Endpoints (for n8n callbacks)
# ============================================================================

class WebhookPayload(BaseModel):
    """Generic webhook payload."""
    event: str
    data: Dict[str, Any]
    callback_url: Optional[str] = None


@app.post("/api/v1/webhook/training-complete", tags=["Webhooks"])
async def training_complete_webhook(payload: WebhookPayload):
    """
    Webhook endpoint for training completion callbacks.

    n8n can register this webhook to receive training completion notifications.
    """
    return {"received": True, "event": payload.event}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
