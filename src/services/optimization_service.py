"""
Optimization Service for Pharma MMM Agent.

Handles budget allocation optimization with constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from scipy.optimize import minimize, differential_evolution
from loguru import logger
import json


# ============================================================================
# Input/Output Contracts
# ============================================================================

class ChannelConstraint(BaseModel):
    """Constraints for a single channel."""
    channel: str
    min_spend: Optional[float] = None  # Absolute minimum
    max_spend: Optional[float] = None  # Absolute maximum
    min_ratio: float = 0.5  # Relative to current (50%)
    max_ratio: float = 2.0  # Relative to current (200%)


class OptimizationRequest(BaseModel):
    """Request for budget optimization."""
    model_id: str
    total_budget: float
    current_allocation: Dict[str, float]  # channel -> spend
    constraints: Optional[List[ChannelConstraint]] = None
    optimization_objective: str = "maximize_sales"  # or "maximize_roi"
    n_periods: int = 52  # Number of periods to optimize for


class OptimizedAllocation(BaseModel):
    """Optimized budget allocation for a channel."""
    channel: str
    current_spend: float
    optimized_spend: float
    change_percentage: float
    expected_contribution: float
    marginal_roi: float


class OptimizationResult(BaseModel):
    """Results from budget optimization."""
    success: bool
    message: str
    total_budget: float
    current_expected_sales: float
    optimized_expected_sales: float
    improvement_percentage: float
    allocations: List[OptimizedAllocation]
    optimization_details: Dict


class ScenarioRequest(BaseModel):
    """Request for scenario analysis."""
    model_id: str
    scenarios: List[Dict[str, float]]  # List of channel -> spend dicts
    scenario_names: Optional[List[str]] = None


class ScenarioResult(BaseModel):
    """Results from a single scenario."""
    scenario_name: str
    total_spend: float
    expected_sales: float
    roi: float
    channel_contributions: Dict[str, float]


class MarginalROIRequest(BaseModel):
    """Request for marginal ROI analysis."""
    model_id: str
    channel: str
    spend_range: Tuple[float, float]
    n_points: int = 20


class MarginalROIResult(BaseModel):
    """Marginal ROI curve for a channel."""
    channel: str
    spend_levels: List[float]
    marginal_roi: List[float]
    total_contribution: List[float]
    optimal_spend: float


# ============================================================================
# Optimization Service Implementation
# ============================================================================

class OptimizationService:
    """
    Budget optimization service for MMM.

    Uses response curves from trained models to optimize
    budget allocation across channels.
    """

    @staticmethod
    def optimize_budget(request: OptimizationRequest) -> OptimizationResult:
        """
        Optimize budget allocation across channels.

        Args:
            request: OptimizationRequest with budget and constraints

        Returns:
            OptimizationResult with optimized allocation
        """
        from .modeling_service import ModelingService

        logger.info(f"Starting budget optimization for total budget: ${request.total_budget:,.0f}")

        # Get model artifacts
        if request.model_id not in ModelingService._models:
            return OptimizationResult(
                success=False,
                message=f"Model {request.model_id} not found",
                total_budget=request.total_budget,
                current_expected_sales=0,
                optimized_expected_sales=0,
                improvement_percentage=0,
                allocations=[],
                optimization_details={}
            )

        model_data = ModelingService._models[request.model_id]
        trace = ModelingService._traces[request.model_id]
        scalers = ModelingService._scalers[request.model_id]
        original_request = model_data['request']

        channels = list(request.current_allocation.keys())
        n_channels = len(channels)

        # Get model parameters (posterior means)
        posterior = trace.posterior
        beta_mean = posterior['beta_spend'].mean(dim=['chain', 'draw']).values
        decay_mean = posterior['adstock_decay'].mean(dim=['chain', 'draw']).values

        if 'sat_k' in posterior:
            sat_k_mean = posterior['sat_k'].mean(dim=['chain', 'draw']).values
            sat_s_mean = posterior['sat_s'].mean(dim=['chain', 'draw']).values
        else:
            sat_k_mean = np.full(n_channels, 0.5)
            sat_s_mean = np.full(n_channels, 1.0)

        # Build response functions
        response_params = {}
        for i, col in enumerate(original_request.spend_columns):
            channel_name = col.replace('_spend', '').replace('_', ' ').title()
            if channel_name in channels:
                scaler = scalers['spend'].get(col, {'mean': 1, 'std': 1})
                response_params[channel_name] = {
                    'beta': beta_mean[i],
                    'decay': decay_mean[i],
                    'sat_k': sat_k_mean[i],
                    'sat_s': sat_s_mean[i],
                    'scaler_mean': scaler['mean'],
                    'scaler_std': scaler['std']
                }

        y_scaler = scalers['y']

        # Define objective function
        def objective(x):
            """Negative expected sales (for minimization)."""
            total_contribution = 0
            for i, channel in enumerate(channels):
                if channel in response_params:
                    params = response_params[channel]
                    spend = x[i]

                    # Scale spend
                    spend_scaled = (spend - params['scaler_mean']) / params['scaler_std']

                    # Saturation
                    x_norm = spend_scaled / (np.abs(spend_scaled) + 1e-8)
                    saturated = np.abs(x_norm) ** params['sat_s'] / (
                        params['sat_k'] ** params['sat_s'] + np.abs(x_norm) ** params['sat_s']
                    )

                    contribution = params['beta'] * saturated * request.n_periods
                    total_contribution += contribution

            # Unscale
            expected_sales = total_contribution * y_scaler['std']

            if request.optimization_objective == "maximize_roi":
                # Maximize ROI instead of absolute sales
                total_spend = np.sum(x)
                return -expected_sales / (total_spend + 1e-8)

            return -expected_sales

        # Set up constraints
        current_spends = np.array([request.current_allocation.get(ch, 0) for ch in channels])

        # Bounds
        bounds = []
        for i, channel in enumerate(channels):
            current = current_spends[i]

            # Check for channel-specific constraints
            min_spend = current * 0.5
            max_spend = current * 2.0

            if request.constraints:
                for constraint in request.constraints:
                    if constraint.channel == channel:
                        if constraint.min_spend is not None:
                            min_spend = constraint.min_spend
                        else:
                            min_spend = current * constraint.min_ratio

                        if constraint.max_spend is not None:
                            max_spend = constraint.max_spend
                        else:
                            max_spend = current * constraint.max_ratio
                        break

            bounds.append((max(0, min_spend), max_spend))

        # Budget constraint
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - request.total_budget}
        ]

        # Initial guess (proportional to current)
        total_current = sum(current_spends)
        if total_current > 0:
            x0 = current_spends * (request.total_budget / total_current)
        else:
            x0 = np.full(n_channels, request.total_budget / n_channels)

        # Ensure x0 is within bounds
        for i, (lb, ub) in enumerate(bounds):
            x0[i] = np.clip(x0[i], lb, ub)

        # Normalize to meet budget constraint
        x0 = x0 * (request.total_budget / x0.sum())

        # Optimize
        logger.info("Running SLSQP optimization...")
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not result.success:
            # Try differential evolution as fallback
            logger.warning("SLSQP failed, trying differential evolution...")

            def de_objective(x):
                # Penalize budget constraint violation
                penalty = 1000 * (np.sum(x) - request.total_budget) ** 2
                return objective(x) + penalty

            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=500,
                seed=42
            )
            result = de_result

        optimized_spends = result.x

        # Normalize to exact budget
        optimized_spends = optimized_spends * (request.total_budget / optimized_spends.sum())

        # Calculate metrics
        current_sales = -objective(current_spends * (request.total_budget / current_spends.sum())
                                   if current_spends.sum() > 0 else current_spends)
        optimized_sales = -objective(optimized_spends)

        improvement = ((optimized_sales - current_sales) / current_sales * 100
                       if current_sales > 0 else 0)

        # Build allocation results
        allocations = []
        for i, channel in enumerate(channels):
            current = current_spends[i]
            optimized = optimized_spends[i]
            change_pct = ((optimized - current) / current * 100) if current > 0 else 0

            # Calculate marginal ROI at optimized spend
            marginal = OptimizationService._calculate_marginal_roi(
                channel, optimized, response_params, y_scaler, request.n_periods
            )

            # Expected contribution
            if channel in response_params:
                params = response_params[channel]
                spend_scaled = (optimized - params['scaler_mean']) / params['scaler_std']
                x_norm = spend_scaled / (np.abs(spend_scaled) + 1e-8)
                saturated = np.abs(x_norm) ** params['sat_s'] / (
                    params['sat_k'] ** params['sat_s'] + np.abs(x_norm) ** params['sat_s']
                )
                contribution = params['beta'] * saturated * request.n_periods * y_scaler['std']
            else:
                contribution = 0

            allocations.append(OptimizedAllocation(
                channel=channel,
                current_spend=float(current),
                optimized_spend=float(optimized),
                change_percentage=float(change_pct),
                expected_contribution=float(contribution),
                marginal_roi=float(marginal)
            ))

        return OptimizationResult(
            success=True,
            message="Optimization completed successfully",
            total_budget=request.total_budget,
            current_expected_sales=float(current_sales),
            optimized_expected_sales=float(optimized_sales),
            improvement_percentage=float(improvement),
            allocations=allocations,
            optimization_details={
                'optimization_method': 'SLSQP',
                'objective': request.optimization_objective,
                'n_iterations': result.nit if hasattr(result, 'nit') else 0,
                'convergence': result.success
            }
        )

    @staticmethod
    def _calculate_marginal_roi(
        channel: str,
        spend: float,
        response_params: Dict,
        y_scaler: Dict,
        n_periods: int,
        delta: float = 1000
    ) -> float:
        """Calculate marginal ROI at a given spend level."""
        if channel not in response_params:
            return 0

        params = response_params[channel]

        def get_contribution(s):
            spend_scaled = (s - params['scaler_mean']) / params['scaler_std']
            x_norm = spend_scaled / (np.abs(spend_scaled) + 1e-8)
            saturated = np.abs(x_norm) ** params['sat_s'] / (
                params['sat_k'] ** params['sat_s'] + np.abs(x_norm) ** params['sat_s']
            )
            return params['beta'] * saturated * n_periods * y_scaler['std']

        contrib_current = get_contribution(spend)
        contrib_delta = get_contribution(spend + delta)

        return (contrib_delta - contrib_current) / delta

    @staticmethod
    def analyze_scenarios(request: ScenarioRequest) -> List[ScenarioResult]:
        """
        Analyze multiple budget scenarios.

        Args:
            request: ScenarioRequest with scenarios to evaluate

        Returns:
            List of ScenarioResult for each scenario
        """
        from .modeling_service import ModelingService

        if request.model_id not in ModelingService._models:
            raise ValueError(f"Model {request.model_id} not found")

        model_data = ModelingService._models[request.model_id]
        trace = ModelingService._traces[request.model_id]
        scalers = ModelingService._scalers[request.model_id]
        original_request = model_data['request']

        posterior = trace.posterior
        beta_mean = posterior['beta_spend'].mean(dim=['chain', 'draw']).values
        decay_mean = posterior['adstock_decay'].mean(dim=['chain', 'draw']).values

        if 'sat_k' in posterior:
            sat_k_mean = posterior['sat_k'].mean(dim=['chain', 'draw']).values
            sat_s_mean = posterior['sat_s'].mean(dim=['chain', 'draw']).values
        else:
            sat_k_mean = np.full(len(original_request.spend_columns), 0.5)
            sat_s_mean = np.full(len(original_request.spend_columns), 1.0)

        y_scaler = scalers['y']

        results = []
        for idx, scenario in enumerate(request.scenarios):
            scenario_name = (request.scenario_names[idx]
                             if request.scenario_names and idx < len(request.scenario_names)
                             else f"Scenario {idx + 1}")

            total_spend = sum(scenario.values())
            channel_contributions = {}
            total_contribution = 0

            for i, col in enumerate(original_request.spend_columns):
                channel_name = col.replace('_spend', '').replace('_', ' ').title()
                spend = scenario.get(channel_name, 0)

                if spend > 0:
                    scaler = scalers['spend'].get(col, {'mean': 1, 'std': 1})
                    spend_scaled = (spend - scaler['mean']) / scaler['std']

                    x_norm = spend_scaled / (np.abs(spend_scaled) + 1e-8)
                    saturated = np.abs(x_norm) ** sat_s_mean[i] / (
                        sat_k_mean[i] ** sat_s_mean[i] + np.abs(x_norm) ** sat_s_mean[i]
                    )

                    contribution = beta_mean[i] * saturated * y_scaler['std']
                    channel_contributions[channel_name] = float(contribution)
                    total_contribution += contribution

            expected_sales = total_contribution
            roi = expected_sales / total_spend if total_spend > 0 else 0

            results.append(ScenarioResult(
                scenario_name=scenario_name,
                total_spend=total_spend,
                expected_sales=expected_sales,
                roi=roi,
                channel_contributions=channel_contributions
            ))

        return results

    @staticmethod
    def get_marginal_roi_curve(request: MarginalROIRequest) -> MarginalROIResult:
        """
        Get marginal ROI curve for a channel.

        Args:
            request: MarginalROIRequest with channel and spend range

        Returns:
            MarginalROIResult with ROI curve data
        """
        from .modeling_service import ModelingService

        if request.model_id not in ModelingService._models:
            raise ValueError(f"Model {request.model_id} not found")

        model_data = ModelingService._models[request.model_id]
        trace = ModelingService._traces[request.model_id]
        scalers = ModelingService._scalers[request.model_id]
        original_request = model_data['request']

        # Find channel index
        channel_idx = None
        channel_col = None
        for i, col in enumerate(original_request.spend_columns):
            channel_name = col.replace('_spend', '').replace('_', ' ').title()
            if channel_name == request.channel:
                channel_idx = i
                channel_col = col
                break

        if channel_idx is None:
            raise ValueError(f"Channel {request.channel} not found")

        posterior = trace.posterior
        beta = float(posterior['beta_spend'].mean(dim=['chain', 'draw']).values[channel_idx])

        if 'sat_k' in posterior:
            sat_k = float(posterior['sat_k'].mean(dim=['chain', 'draw']).values[channel_idx])
            sat_s = float(posterior['sat_s'].mean(dim=['chain', 'draw']).values[channel_idx])
        else:
            sat_k = 0.5
            sat_s = 1.0

        scaler = scalers['spend'].get(channel_col, {'mean': 1, 'std': 1})
        y_scaler = scalers['y']

        # Generate spend levels
        spend_levels = np.linspace(request.spend_range[0], request.spend_range[1], request.n_points)

        contributions = []
        marginal_rois = []

        for spend in spend_levels:
            spend_scaled = (spend - scaler['mean']) / scaler['std']
            x_norm = spend_scaled / (np.abs(spend_scaled) + 1e-8)
            saturated = np.abs(x_norm) ** sat_s / (sat_k ** sat_s + np.abs(x_norm) ** sat_s)
            contribution = beta * saturated * y_scaler['std']
            contributions.append(contribution)

        # Calculate marginal ROI (derivative)
        for i in range(len(spend_levels)):
            if i == 0:
                marginal = (contributions[1] - contributions[0]) / (spend_levels[1] - spend_levels[0])
            elif i == len(spend_levels) - 1:
                marginal = (contributions[-1] - contributions[-2]) / (spend_levels[-1] - spend_levels[-2])
            else:
                marginal = (contributions[i+1] - contributions[i-1]) / (spend_levels[i+1] - spend_levels[i-1])
            marginal_rois.append(marginal)

        # Find optimal spend (where marginal ROI = 1)
        optimal_idx = np.argmin(np.abs(np.array(marginal_rois) - 1))
        optimal_spend = spend_levels[optimal_idx]

        return MarginalROIResult(
            channel=request.channel,
            spend_levels=spend_levels.tolist(),
            marginal_roi=marginal_rois,
            total_contribution=contributions,
            optimal_spend=float(optimal_spend)
        )
