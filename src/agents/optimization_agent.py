"""
Optimization Agent for Pharma MMM Agent.

Coordinates budget optimization and scenario analysis tasks.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel
from loguru import logger

from ..services.optimization_service import (
    OptimizationService, OptimizationRequest, OptimizationResult,
    ScenarioRequest, ScenarioResult, MarginalROIRequest, MarginalROIResult,
    ChannelConstraint
)


class OptimizationAgentRequest(BaseModel):
    """Request for Optimization Agent operations."""
    operation: str  # "optimize", "scenarios", "marginal_roi"
    model_id: str
    # Optimization params
    total_budget: Optional[float] = None
    current_allocation: Optional[Dict[str, float]] = None
    constraints: Optional[List[Dict]] = None
    optimization_objective: str = "maximize_sales"
    n_periods: int = 52
    # Scenario params
    scenarios: Optional[List[Dict[str, float]]] = None
    scenario_names: Optional[List[str]] = None
    # Marginal ROI params
    channel: Optional[str] = None
    spend_range: Optional[tuple] = None


class OptimizationAgentResponse(BaseModel):
    """Response from Optimization Agent operations."""
    success: bool
    operation: str
    message: str
    optimization_result: Optional[OptimizationResult] = None
    scenario_results: Optional[List[ScenarioResult]] = None
    marginal_roi_result: Optional[MarginalROIResult] = None


class OptimizationAgent:
    """
    Agent for handling optimization operations.

    Coordinates the OptimizationService to optimize budgets,
    analyze scenarios, and calculate marginal ROI.
    """

    def __init__(self):
        self.service = OptimizationService()

    def execute(self, request: OptimizationAgentRequest) -> OptimizationAgentResponse:
        """Execute an optimization operation."""
        logger.info(f"OptimizationAgent executing operation: {request.operation}")

        if request.operation == "optimize":
            return self._optimize(request)
        elif request.operation == "scenarios":
            return self._scenarios(request)
        elif request.operation == "marginal_roi":
            return self._marginal_roi(request)
        else:
            return OptimizationAgentResponse(
                success=False,
                operation=request.operation,
                message=f"Unknown operation: {request.operation}"
            )

    def _optimize(self, request: OptimizationAgentRequest) -> OptimizationAgentResponse:
        """Optimize budget allocation."""
        try:
            if not request.total_budget or not request.current_allocation:
                return OptimizationAgentResponse(
                    success=False,
                    operation="optimize",
                    message="Missing total_budget or current_allocation"
                )

            # Build constraints if provided
            constraints = None
            if request.constraints:
                constraints = [
                    ChannelConstraint(**c) for c in request.constraints
                ]

            opt_request = OptimizationRequest(
                model_id=request.model_id,
                total_budget=request.total_budget,
                current_allocation=request.current_allocation,
                constraints=constraints,
                optimization_objective=request.optimization_objective,
                n_periods=request.n_periods
            )

            result = OptimizationService.optimize_budget(opt_request)

            return OptimizationAgentResponse(
                success=result.success,
                operation="optimize",
                message=result.message,
                optimization_result=result
            )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return OptimizationAgentResponse(
                success=False,
                operation="optimize",
                message=f"Optimization failed: {str(e)}"
            )

    def _scenarios(self, request: OptimizationAgentRequest) -> OptimizationAgentResponse:
        """Analyze multiple scenarios."""
        try:
            if not request.scenarios:
                return OptimizationAgentResponse(
                    success=False,
                    operation="scenarios",
                    message="No scenarios provided"
                )

            scenario_request = ScenarioRequest(
                model_id=request.model_id,
                scenarios=request.scenarios,
                scenario_names=request.scenario_names
            )

            results = OptimizationService.analyze_scenarios(scenario_request)

            return OptimizationAgentResponse(
                success=True,
                operation="scenarios",
                message=f"Analyzed {len(results)} scenarios",
                scenario_results=results
            )

        except Exception as e:
            logger.error(f"Scenario analysis failed: {e}")
            return OptimizationAgentResponse(
                success=False,
                operation="scenarios",
                message=f"Scenario analysis failed: {str(e)}"
            )

    def _marginal_roi(self, request: OptimizationAgentRequest) -> OptimizationAgentResponse:
        """Calculate marginal ROI curve."""
        try:
            if not request.channel or not request.spend_range:
                return OptimizationAgentResponse(
                    success=False,
                    operation="marginal_roi",
                    message="Missing channel or spend_range"
                )

            roi_request = MarginalROIRequest(
                model_id=request.model_id,
                channel=request.channel,
                spend_range=request.spend_range
            )

            result = OptimizationService.get_marginal_roi_curve(roi_request)

            return OptimizationAgentResponse(
                success=True,
                operation="marginal_roi",
                message=f"Marginal ROI calculated for {request.channel}",
                marginal_roi_result=result
            )

        except Exception as e:
            logger.error(f"Marginal ROI calculation failed: {e}")
            return OptimizationAgentResponse(
                success=False,
                operation="marginal_roi",
                message=f"Marginal ROI calculation failed: {str(e)}"
            )
