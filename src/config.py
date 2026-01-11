"""Configuration settings for Pharma MMM Agent."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class ChannelConfig(BaseModel):
    """Configuration for a marketing channel."""
    name: str
    adstock_type: str = "geometric"  # geometric or weibull
    adstock_decay_min: float = 0.3
    adstock_decay_max: float = 0.9
    saturation_type: str = "hill"
    typical_spend_range: tuple = (100000, 5000000)
    carryover_weeks: int = 8


class PharmaChannels:
    """Pharma-specific marketing channels with longer carryover effects."""

    DTC_TV = ChannelConfig(
        name="DTC TV",
        adstock_decay_min=0.5,
        adstock_decay_max=0.85,
        carryover_weeks=10,
        typical_spend_range=(500000, 10000000)
    )

    DTC_DIGITAL = ChannelConfig(
        name="DTC Digital",
        adstock_decay_min=0.3,
        adstock_decay_max=0.7,
        carryover_weeks=6,
        typical_spend_range=(100000, 3000000)
    )

    HCP_DETAILING = ChannelConfig(
        name="HCP Detailing",
        adstock_decay_min=0.6,
        adstock_decay_max=0.9,
        carryover_weeks=12,
        typical_spend_range=(200000, 5000000)
    )

    CONFERENCES = ChannelConfig(
        name="Conferences",
        adstock_decay_min=0.4,
        adstock_decay_max=0.8,
        carryover_weeks=8,
        typical_spend_range=(50000, 2000000)
    )

    SAMPLES = ChannelConfig(
        name="Samples",
        adstock_decay_min=0.5,
        adstock_decay_max=0.85,
        carryover_weeks=10,
        typical_spend_range=(100000, 4000000)
    )

    JOURNAL_ADS = ChannelConfig(
        name="Journal Ads",
        adstock_decay_min=0.4,
        adstock_decay_max=0.75,
        carryover_weeks=8,
        typical_spend_range=(50000, 1500000)
    )

    @classmethod
    def get_all(cls) -> List[ChannelConfig]:
        return [
            cls.DTC_TV, cls.DTC_DIGITAL, cls.HCP_DETAILING,
            cls.CONFERENCES, cls.SAMPLES, cls.JOURNAL_ADS
        ]

    @classmethod
    def get_names(cls) -> List[str]:
        return [ch.name for ch in cls.get_all()]


class ModelConfig(BaseModel):
    """Configuration for the Bayesian MMM model."""

    # Adstock parameters
    adstock_max_lag: int = 12  # weeks

    # Saturation (Hill function) parameters
    hill_k_min: float = 0.5
    hill_k_max: float = 3.0
    hill_s_min: float = 0.1
    hill_s_max: float = 1.0

    # MCMC parameters
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 2
    target_accept: float = 0.9

    # Priors
    intercept_mu: float = 0.0
    intercept_sigma: float = 1.0
    beta_mu: float = 0.0
    beta_sigma: float = 0.5


class OptimizationConfig(BaseModel):
    """Configuration for budget optimization."""

    min_budget_ratio: float = 0.5  # Min 50% of current spend
    max_budget_ratio: float = 2.0  # Max 200% of current spend
    total_budget_tolerance: float = 0.01  # 1% tolerance
    optimization_method: str = "SLSQP"


class AppConfig(BaseModel):
    """Main application configuration."""

    # API Keys
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Server settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    # Data settings
    data_path: str = "data"

    # Model settings
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


# Global config instance
config = AppConfig()
