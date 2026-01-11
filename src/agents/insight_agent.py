"""
Insight Agent for Pharma MMM Agent.

Coordinates AI-powered insight generation using Claude API.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel
from loguru import logger

from ..services.insight_service import (
    InsightService, InsightRequest, InsightResponse,
    QARequest, QAResponse, ReportRequest, ReportResponse
)


class InsightAgentRequest(BaseModel):
    """Request for Insight Agent operations."""
    operation: str  # "generate", "qa", "report"
    model_results_json: str
    # Generation params
    insight_type: str = "summary"
    focus_areas: Optional[List[str]] = None
    # QA params
    question: Optional[str] = None
    context: Optional[str] = None
    # Report params
    optimization_results_json: Optional[str] = None
    include_sections: List[str] = ["executive_summary", "channel_performance", "recommendations"]


class InsightAgentResponse(BaseModel):
    """Response from Insight Agent operations."""
    success: bool
    operation: str
    message: str
    insight: Optional[InsightResponse] = None
    qa_response: Optional[QAResponse] = None
    report: Optional[ReportResponse] = None


class InsightAgent:
    """
    Agent for handling insight generation.

    Coordinates the InsightService to generate summaries,
    answer questions, and create reports using Claude API.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.service = InsightService(api_key)

    def execute(self, request: InsightAgentRequest) -> InsightAgentResponse:
        """Execute an insight operation."""
        logger.info(f"InsightAgent executing operation: {request.operation}")

        if request.operation == "generate":
            return self._generate(request)
        elif request.operation == "qa":
            return self._qa(request)
        elif request.operation == "report":
            return self._report(request)
        else:
            return InsightAgentResponse(
                success=False,
                operation=request.operation,
                message=f"Unknown operation: {request.operation}"
            )

    def _generate(self, request: InsightAgentRequest) -> InsightAgentResponse:
        """Generate insights from model results."""
        try:
            insight_request = InsightRequest(
                model_results_json=request.model_results_json,
                insight_type=request.insight_type,
                focus_areas=request.focus_areas
            )

            result = self.service.generate_insight(insight_request)

            return InsightAgentResponse(
                success=True,
                operation="generate",
                message=f"Generated {request.insight_type} insight",
                insight=result
            )

        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return InsightAgentResponse(
                success=False,
                operation="generate",
                message=f"Insight generation failed: {str(e)}"
            )

    def _qa(self, request: InsightAgentRequest) -> InsightAgentResponse:
        """Answer a question about model results."""
        try:
            if not request.question:
                return InsightAgentResponse(
                    success=False,
                    operation="qa",
                    message="No question provided"
                )

            qa_request = QARequest(
                question=request.question,
                model_results_json=request.model_results_json,
                context=request.context
            )

            result = self.service.answer_question(qa_request)

            return InsightAgentResponse(
                success=True,
                operation="qa",
                message="Question answered",
                qa_response=result
            )

        except Exception as e:
            logger.error(f"QA failed: {e}")
            return InsightAgentResponse(
                success=False,
                operation="qa",
                message=f"QA failed: {str(e)}"
            )

    def _report(self, request: InsightAgentRequest) -> InsightAgentResponse:
        """Generate a comprehensive report."""
        try:
            report_request = ReportRequest(
                model_results_json=request.model_results_json,
                optimization_results_json=request.optimization_results_json,
                include_sections=request.include_sections
            )

            result = self.service.generate_report(report_request)

            return InsightAgentResponse(
                success=True,
                operation="report",
                message="Report generated",
                report=result
            )

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return InsightAgentResponse(
                success=False,
                operation="report",
                message=f"Report generation failed: {str(e)}"
            )

    def quick_summary(self, channel_contributions: List[Dict]) -> str:
        """
        Generate a quick channel summary without API call.

        Useful for fast feedback or when API is unavailable.
        """
        return InsightService.generate_channel_summary(channel_contributions)
