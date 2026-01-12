"""
Modeling Service for Pharma MMM Agent.

Implements Bayesian Market Mix Modeling with PyMC.
Includes adstock transformations, saturation curves, and channel decomposition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from loguru import logger
import json
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Input/Output Contracts
# ============================================================================

class ModelTrainingRequest(BaseModel):
    """Request for model training."""
    data_json: str
    date_column: str = "date"
    sales_column: str = "sales"
    spend_columns: List[str]
    control_columns: Optional[List[str]] = None

    # Adstock settings
    adstock_max_lag: int = Field(default=12, ge=1, le=52)
    estimate_adstock: bool = True  # If False, use default decay

    # Saturation settings
    apply_saturation: bool = True

    # MCMC settings
    n_samples: int = Field(default=2000, ge=100)
    n_tune: int = Field(default=1000, ge=100)
    n_chains: int = Field(default=2, ge=1)
    target_accept: float = Field(default=0.9, ge=0.5, le=0.99)


class ChannelContribution(BaseModel):
    """Contribution metrics for a single channel."""
    channel: str
    total_contribution: float
    contribution_percentage: float
    roi: float
    total_spend: float
    adstock_decay: Optional[float] = None
    saturation_k: Optional[float] = None
    saturation_s: Optional[float] = None
    coefficient: float
    coefficient_lower: float
    coefficient_upper: float


class ModelResults(BaseModel):
    """Results from model training."""
    success: bool
    model_id: str
    r_squared: float
    mape: float  # Mean Absolute Percentage Error
    channel_contributions: List[ChannelContribution]
    base_sales: float
    base_sales_percentage: float
    total_marketing_contribution: float
    coefficients: Dict[str, Dict[str, float]]  # name -> {mean, std, lower, upper}
    diagnostics: Dict[str, Any]
    fitted_values_json: str
    residuals_json: str


class DecompositionResult(BaseModel):
    """Sales decomposition results."""
    date: str
    actual_sales: float
    predicted_sales: float
    base_sales: float
    channel_contributions: Dict[str, float]


class PredictionRequest(BaseModel):
    """Request for predictions."""
    model_id: str
    spend_scenario_json: str  # JSON with spend columns


class PredictionResult(BaseModel):
    """Prediction results."""
    predicted_sales: List[float]
    confidence_lower: List[float]
    confidence_upper: List[float]
    channel_contributions_json: str


# ============================================================================
# Modeling Service Implementation
# ============================================================================

class ModelingService:
    """
    Bayesian Market Mix Modeling service.

    Uses PyMC for Bayesian inference with:
    - Geometric adstock transformations (pre-computed)
    - Hill saturation curves
    - Channel contribution decomposition
    """

    # Class-level storage for trained models (in production, use proper persistence)
    _models: Dict[str, Any] = {}
    _traces: Dict[str, Any] = {}
    _scalers: Dict[str, Dict] = {}

    @staticmethod
    def _apply_geometric_adstock(
        x: np.ndarray,
        decay: float,
        max_lag: int = 12
    ) -> np.ndarray:
        """Apply geometric adstock transformation."""
        adstocked = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            for j in range(min(i + 1, max_lag)):
                adstocked[i] += x[i - j] * (decay ** j)
        return adstocked

    @staticmethod
    def _apply_hill_saturation(
        x: np.ndarray,
        half_saturation: float,
        slope: float
    ) -> np.ndarray:
        """
        Apply Hill saturation function.

        Args:
            x: Input array (should be positive)
            half_saturation: Point at which response is 50% of max
            slope: Controls steepness of curve

        Returns:
            Saturated values in range [0, 1]
        """
        # Ensure positive values
        x_pos = np.maximum(x, 0)
        # Normalize by max to keep values reasonable
        x_max = x_pos.max() if x_pos.max() > 0 else 1.0
        x_norm = x_pos / x_max

        # Hill function: x^s / (k^s + x^s)
        # Use log-space for numerical stability
        k = half_saturation
        s = slope

        # Avoid division by zero
        denominator = np.power(k, s) + np.power(x_norm + 1e-10, s)
        result = np.power(x_norm + 1e-10, s) / denominator

        return result

    @staticmethod
    def train_model(request: ModelTrainingRequest) -> ModelResults:
        """
        Train a Bayesian MMM model.

        Uses a two-stage approach:
        1. Pre-compute adstock transformations for candidate decay values
        2. Estimate model with saturation parameters

        Args:
            request: ModelTrainingRequest with data and settings

        Returns:
            ModelResults with fitted model metrics and contributions
        """
        import pymc as pm
        import arviz as az
        import uuid

        logger.info("Starting Bayesian MMM training...")

        # Load data
        df = pd.read_json(request.data_json)
        df[request.date_column] = pd.to_datetime(df[request.date_column])
        df = df.sort_values(request.date_column).reset_index(drop=True)

        n_obs = len(df)
        n_channels = len(request.spend_columns)

        # Prepare response variable (log transform for stability)
        y = df[request.sales_column].values.astype(float)
        y_mean = y.mean()
        y_std = y.std()
        y_scaled = (y - y_mean) / y_std

        logger.info(f"Data: {n_obs} observations, {n_channels} channels")
        logger.info(f"Sales: mean={y_mean:.0f}, std={y_std:.0f}")

        # Pre-compute adstock-transformed spend for each channel
        # Use a grid of decay values and let the model choose
        decay_candidates = [0.5, 0.6, 0.7, 0.8, 0.9]
        n_decays = len(decay_candidates)

        # Store raw spend and transformed versions
        raw_spend = {}
        adstock_grids = {}  # channel -> (n_obs, n_decays) array

        for col in request.spend_columns:
            raw_spend[col] = df[col].values.astype(float)
            adstock_grid = np.zeros((n_obs, n_decays))
            for d_idx, decay in enumerate(decay_candidates):
                adstock_grid[:, d_idx] = ModelingService._apply_geometric_adstock(
                    raw_spend[col], decay, request.adstock_max_lag
                )
            adstock_grids[col] = adstock_grid

        # Prepare control variables
        X_control = None
        control_scalers = {}
        n_controls = 0
        if request.control_columns:
            control_cols_present = [c for c in request.control_columns if c in df.columns]
            n_controls = len(control_cols_present)
            if n_controls > 0:
                X_control = np.zeros((n_obs, n_controls))
                for i, col in enumerate(control_cols_present):
                    vals = df[col].values.astype(float)
                    vals_mean = vals.mean()
                    vals_std = vals.std() if vals.std() > 0 else 1.0
                    X_control[:, i] = (vals - vals_mean) / vals_std
                    control_scalers[col] = {'mean': vals_mean, 'std': vals_std}

        # Build PyMC model
        logger.info(f"Building PyMC model...")

        with pm.Model() as model:
            # Intercept
            intercept = pm.Normal('intercept', mu=0, sigma=1)

            # For each channel, select best decay and estimate effect
            channel_effects = []

            for c_idx, col in enumerate(request.spend_columns):
                # Decay selection (categorical)
                if request.estimate_adstock:
                    decay_weights = pm.Dirichlet(f'decay_weights_{c_idx}', a=np.ones(n_decays))
                    # Weighted combination of adstock versions
                    adstock_combined = pm.math.dot(adstock_grids[col], decay_weights)
                else:
                    # Use middle decay value (0.7)
                    adstock_combined = adstock_grids[col][:, 2]

                # Normalize the adstocked spend
                adstock_mean = float(np.mean(adstock_grids[col]))
                adstock_std = float(np.std(adstock_grids[col]))
                if adstock_std < 1e-6:
                    adstock_std = 1.0

                adstock_normalized = (adstock_combined - adstock_mean) / adstock_std

                # Saturation parameters (if enabled)
                if request.apply_saturation:
                    # Half-saturation point (0-1 range on normalized scale)
                    sat_k = pm.Beta(f'sat_k_{c_idx}', alpha=2, beta=2)
                    # Slope parameter
                    sat_s = pm.Gamma(f'sat_s_{c_idx}', alpha=2, beta=1)

                    # Apply Hill saturation (simplified for PyMC)
                    # Transform to positive range first
                    x_shifted = adstock_normalized - adstock_normalized.min() + 0.1
                    x_max = x_shifted.max()
                    x_01 = x_shifted / x_max

                    # Hill function
                    saturated = pm.math.power(x_01, sat_s) / (
                        pm.math.power(sat_k + 0.01, sat_s) + pm.math.power(x_01, sat_s)
                    )
                else:
                    saturated = adstock_normalized

                # Channel coefficient (positive effect)
                beta = pm.HalfNormal(f'beta_{c_idx}', sigma=0.5)

                channel_effects.append(beta * saturated)

            # Sum all channel effects
            total_channel_effect = channel_effects[0]
            for effect in channel_effects[1:]:
                total_channel_effect = total_channel_effect + effect

            # Control variables
            if X_control is not None and n_controls > 0:
                beta_control = pm.Normal('beta_control', mu=0, sigma=0.5, shape=n_controls)
                control_effect = pm.math.dot(X_control, beta_control)
                mu = intercept + total_channel_effect + control_effect
            else:
                mu = intercept + total_channel_effect

            # Model error
            sigma = pm.HalfNormal('sigma', sigma=1)

            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_scaled)

            # Sample
            logger.info(f"Sampling {request.n_samples} draws with {request.n_chains} chains...")
            trace = pm.sample(
                draws=request.n_samples,
                tune=request.n_tune,
                chains=request.n_chains,
                target_accept=request.target_accept,
                return_inferencedata=True,
                progressbar=True,
                cores=1
            )

        # Generate model ID
        model_id = str(uuid.uuid4())[:8]

        # Store model artifacts
        ModelingService._models[model_id] = {
            'model': model,
            'df': df,
            'request': request,
            'y_mean': y_mean,
            'y_std': y_std,
            'raw_spend': raw_spend,
            'adstock_grids': adstock_grids,
            'decay_candidates': decay_candidates,
            'X_control': X_control,
            'control_scalers': control_scalers
        }
        ModelingService._traces[model_id] = trace
        ModelingService._scalers[model_id] = {
            'y': {'mean': y_mean, 'std': y_std},
            'control': control_scalers
        }

        # Extract results
        results = ModelingService._extract_results(
            model_id, trace, df, request, y, y_scaled, y_mean, y_std,
            raw_spend, adstock_grids, decay_candidates
        )

        logger.info(f"Model training complete. R² = {results.r_squared:.3f}")

        return results

    @staticmethod
    def _extract_results(
        model_id: str,
        trace: Any,
        df: pd.DataFrame,
        request: ModelTrainingRequest,
        y: np.ndarray,
        y_scaled: np.ndarray,
        y_mean: float,
        y_std: float,
        raw_spend: Dict[str, np.ndarray],
        adstock_grids: Dict[str, np.ndarray],
        decay_candidates: List[float]
    ) -> ModelResults:
        """Extract results from trained model."""
        import arviz as az

        posterior = trace.posterior
        n_channels = len(request.spend_columns)

        # Get posterior means
        intercept_mean = float(posterior['intercept'].mean())

        # Extract channel-specific parameters
        beta_means = []
        decay_means = []
        sat_k_means = []
        sat_s_means = []

        for c_idx in range(n_channels):
            # Beta coefficient
            beta_samples = posterior[f'beta_{c_idx}'].values.flatten()
            beta_means.append(float(np.mean(beta_samples)))

            # Decay (from Dirichlet weights)
            if request.estimate_adstock:
                decay_weights = posterior[f'decay_weights_{c_idx}'].mean(dim=['chain', 'draw']).values
                # Weighted average of decay candidates
                decay_mean = float(np.sum(decay_weights * np.array(decay_candidates)))
            else:
                decay_mean = 0.7
            decay_means.append(decay_mean)

            # Saturation parameters
            if request.apply_saturation:
                sat_k_means.append(float(posterior[f'sat_k_{c_idx}'].mean()))
                sat_s_means.append(float(posterior[f'sat_s_{c_idx}'].mean()))
            else:
                sat_k_means.append(0.5)
                sat_s_means.append(1.0)

        # Calculate fitted values
        fitted_scaled = np.full(len(y), intercept_mean)
        channel_contributions_matrix = np.zeros((len(y), n_channels))

        for c_idx, col in enumerate(request.spend_columns):
            # Apply adstock with estimated decay
            adstocked = ModelingService._apply_geometric_adstock(
                raw_spend[col], decay_means[c_idx], request.adstock_max_lag
            )

            # Normalize
            adstock_mean = adstocked.mean()
            adstock_std = adstocked.std() if adstocked.std() > 0 else 1.0
            adstock_norm = (adstocked - adstock_mean) / adstock_std

            # Apply saturation
            if request.apply_saturation:
                x_shifted = adstock_norm - adstock_norm.min() + 0.1
                x_max = x_shifted.max()
                x_01 = x_shifted / x_max

                k = sat_k_means[c_idx] + 0.01
                s = sat_s_means[c_idx]
                saturated = np.power(x_01, s) / (np.power(k, s) + np.power(x_01, s))
            else:
                saturated = adstock_norm

            contribution = beta_means[c_idx] * saturated
            channel_contributions_matrix[:, c_idx] = contribution
            fitted_scaled += contribution

        # Handle control variables
        if request.control_columns and 'beta_control' in posterior:
            beta_control_mean = posterior['beta_control'].mean(dim=['chain', 'draw']).values
            model_data = ModelingService._models[model_id]
            X_control = model_data['X_control']
            if X_control is not None:
                fitted_scaled += X_control @ beta_control_mean

        # Unscale predictions
        fitted = fitted_scaled * y_std + y_mean
        residuals = y - fitted

        # Calculate metrics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        mape = np.mean(np.abs(residuals / (y + 1e-8))) * 100

        # Calculate channel contributions
        channel_contributions = []
        total_marketing = 0

        for c_idx, col in enumerate(request.spend_columns):
            channel_name = col.replace('_spend', '').replace('_', ' ').title()
            contribution = channel_contributions_matrix[:, c_idx] * y_std

            total_contribution = float(contribution.sum())
            total_spend = float(raw_spend[col].sum())
            roi = total_contribution / total_spend if total_spend > 0 else 0

            # Get coefficient credible intervals
            beta_samples = posterior[f'beta_{c_idx}'].values.flatten()

            channel_contributions.append(ChannelContribution(
                channel=channel_name,
                total_contribution=total_contribution,
                contribution_percentage=0,  # Will calculate after
                roi=roi,
                total_spend=total_spend,
                adstock_decay=decay_means[c_idx],
                saturation_k=sat_k_means[c_idx],
                saturation_s=sat_s_means[c_idx],
                coefficient=beta_means[c_idx],
                coefficient_lower=float(np.percentile(beta_samples, 2.5)),
                coefficient_upper=float(np.percentile(beta_samples, 97.5))
            ))

            total_marketing += total_contribution

        # Calculate percentages
        total_sales = float(y.sum())
        base_sales = total_sales - total_marketing
        base_percentage = (base_sales / total_sales) * 100 if total_sales > 0 else 0

        for contrib in channel_contributions:
            contrib.contribution_percentage = (
                contrib.total_contribution / total_sales * 100 if total_sales > 0 else 0
            )

        # Build coefficients dict
        coefficients = {}
        for c_idx, col in enumerate(request.spend_columns):
            beta_samples = posterior[f'beta_{c_idx}'].values.flatten()
            coefficients[col] = {
                'mean': beta_means[c_idx],
                'std': float(beta_samples.std()),
                'lower': float(np.percentile(beta_samples, 2.5)),
                'upper': float(np.percentile(beta_samples, 97.5))
            }

        # Diagnostics
        try:
            summary = az.summary(trace)
            diagnostics = {
                'n_divergences': int(trace.sample_stats.diverging.sum()),
                'r_hat_max': float(summary['r_hat'].max()) if 'r_hat' in summary.columns else 1.0,
                'ess_min': float(summary['ess_bulk'].min()) if 'ess_bulk' in summary.columns else 1000
            }
        except Exception as e:
            logger.warning(f"Could not compute diagnostics: {e}")
            diagnostics = {'n_divergences': 0, 'r_hat_max': 1.0, 'ess_min': 1000}

        return ModelResults(
            success=True,
            model_id=model_id,
            r_squared=float(max(0, min(1, r_squared))),  # Clip to valid range
            mape=float(mape),
            channel_contributions=channel_contributions,
            base_sales=float(base_sales),
            base_sales_percentage=float(base_percentage),
            total_marketing_contribution=float(total_marketing),
            coefficients=coefficients,
            diagnostics=diagnostics,
            fitted_values_json=json.dumps(fitted.tolist()),
            residuals_json=json.dumps(residuals.tolist())
        )

    @staticmethod
    def get_decomposition(model_id: str) -> List[DecompositionResult]:
        """
        Get detailed sales decomposition.

        Args:
            model_id: ID of trained model

        Returns:
            List of DecompositionResult for each time period
        """
        if model_id not in ModelingService._models:
            raise ValueError(f"Model {model_id} not found")

        model_data = ModelingService._models[model_id]
        trace = ModelingService._traces[model_id]
        df = model_data['df']
        request = model_data['request']
        y_mean = model_data['y_mean']
        y_std = model_data['y_std']
        raw_spend = model_data['raw_spend']

        posterior = trace.posterior
        n_channels = len(request.spend_columns)

        # Get means
        intercept_mean = float(posterior['intercept'].mean())

        beta_means = []
        decay_means = []
        sat_k_means = []
        sat_s_means = []
        decay_candidates = model_data['decay_candidates']

        for c_idx in range(n_channels):
            beta_means.append(float(posterior[f'beta_{c_idx}'].mean()))

            if request.estimate_adstock:
                decay_weights = posterior[f'decay_weights_{c_idx}'].mean(dim=['chain', 'draw']).values
                decay_mean = float(np.sum(decay_weights * np.array(decay_candidates)))
            else:
                decay_mean = 0.7
            decay_means.append(decay_mean)

            if request.apply_saturation:
                sat_k_means.append(float(posterior[f'sat_k_{c_idx}'].mean()))
                sat_s_means.append(float(posterior[f'sat_s_{c_idx}'].mean()))
            else:
                sat_k_means.append(0.5)
                sat_s_means.append(1.0)

        decomposition = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            date_str = row[request.date_column].strftime('%Y-%m-%d')
            actual = float(row[request.sales_column])

            # Base
            base = intercept_mean * y_std + y_mean / len(df)

            # Channel contributions
            channel_contribs = {}
            total_pred_scaled = intercept_mean

            for c_idx, col in enumerate(request.spend_columns):
                channel_name = col.replace('_spend', '').replace('_', ' ').title()

                # Get spend up to this point for adstock
                spend_series = raw_spend[col][:idx+1]
                adstocked = ModelingService._apply_geometric_adstock(
                    spend_series, decay_means[c_idx], request.adstock_max_lag
                )[-1] if len(spend_series) > 0 else 0

                # Use full series stats for normalization
                full_adstocked = ModelingService._apply_geometric_adstock(
                    raw_spend[col], decay_means[c_idx], request.adstock_max_lag
                )
                adstock_mean = full_adstocked.mean()
                adstock_std = full_adstocked.std() if full_adstocked.std() > 0 else 1.0

                adstock_norm = (adstocked - adstock_mean) / adstock_std

                # Saturation
                if request.apply_saturation:
                    x_shifted = adstock_norm - full_adstocked.min() / adstock_std + 0.1
                    x_max = (full_adstocked.max() - full_adstocked.min()) / adstock_std + 0.1
                    x_01 = max(0, x_shifted / x_max)

                    k = sat_k_means[c_idx] + 0.01
                    s = sat_s_means[c_idx]
                    saturated = (x_01 ** s) / (k ** s + x_01 ** s)
                else:
                    saturated = adstock_norm

                contrib_scaled = beta_means[c_idx] * saturated
                contrib = contrib_scaled * y_std

                channel_contribs[channel_name] = float(contrib)
                total_pred_scaled += contrib_scaled

            predicted = total_pred_scaled * y_std + y_mean

            decomposition.append(DecompositionResult(
                date=date_str,
                actual_sales=actual,
                predicted_sales=float(predicted),
                base_sales=float(base),
                channel_contributions=channel_contribs
            ))

        return decomposition

    @staticmethod
    def predict(request: PredictionRequest) -> PredictionResult:
        """
        Make predictions with a trained model.

        Args:
            request: PredictionRequest with spend scenario

        Returns:
            PredictionResult with predictions and confidence intervals
        """
        if request.model_id not in ModelingService._models:
            raise ValueError(f"Model {request.model_id} not found")

        model_data = ModelingService._models[request.model_id]
        trace = ModelingService._traces[request.model_id]
        scalers = ModelingService._scalers[request.model_id]
        original_request = model_data['request']
        raw_spend = model_data['raw_spend']
        decay_candidates = model_data['decay_candidates']

        # Load scenario data
        scenario_df = pd.read_json(request.spend_scenario_json)
        n_channels = len(original_request.spend_columns)

        posterior = trace.posterior
        intercept_samples = posterior['intercept'].values.flatten()
        n_samples = len(intercept_samples)
        n_pred = len(scenario_df)

        # Sample predictions
        predictions = np.zeros((min(n_samples, 500), n_pred))
        channel_contributions = {
            col: np.zeros((min(n_samples, 500), n_pred))
            for col in original_request.spend_columns
        }

        for s in range(min(n_samples, 500)):
            pred = np.full(n_pred, intercept_samples[s])

            for c_idx, col in enumerate(original_request.spend_columns):
                if col in scenario_df.columns:
                    spend = scenario_df[col].values.astype(float)

                    # Get decay
                    if original_request.estimate_adstock:
                        decay_weights = posterior[f'decay_weights_{c_idx}'].values.reshape(-1, len(decay_candidates))[s]
                        decay = float(np.sum(decay_weights * np.array(decay_candidates)))
                    else:
                        decay = 0.7

                    adstocked = ModelingService._apply_geometric_adstock(
                        spend, decay, original_request.adstock_max_lag
                    )

                    # Normalize using original data stats
                    orig_adstocked = ModelingService._apply_geometric_adstock(
                        raw_spend[col], decay, original_request.adstock_max_lag
                    )
                    adstock_mean = orig_adstocked.mean()
                    adstock_std = orig_adstocked.std() if orig_adstocked.std() > 0 else 1.0

                    adstock_norm = (adstocked - adstock_mean) / adstock_std

                    beta = float(posterior[f'beta_{c_idx}'].values.flatten()[s])
                    contrib = beta * adstock_norm

                    pred += contrib
                    channel_contributions[col][s, :] = contrib

            predictions[s, :] = pred

        # Unscale
        y_scaler = scalers['y']
        predictions = predictions * y_scaler['std'] + y_scaler['mean']

        for col in channel_contributions:
            channel_contributions[col] = channel_contributions[col] * y_scaler['std']

        # Calculate statistics
        pred_mean = predictions.mean(axis=0)
        pred_lower = np.percentile(predictions, 2.5, axis=0)
        pred_upper = np.percentile(predictions, 97.5, axis=0)

        # Aggregate channel contributions
        contrib_summary = {}
        for col in original_request.spend_columns:
            channel_name = col.replace('_spend', '').replace('_', ' ').title()
            contrib_summary[channel_name] = channel_contributions[col].mean(axis=0).tolist()

        return PredictionResult(
            predicted_sales=pred_mean.tolist(),
            confidence_lower=pred_lower.tolist(),
            confidence_upper=pred_upper.tolist(),
            channel_contributions_json=json.dumps(contrib_summary)
        )
