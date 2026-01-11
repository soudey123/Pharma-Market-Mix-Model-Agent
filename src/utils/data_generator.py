"""
Sample Data Generator for Pharma Market Mix Modeling.

Generates realistic 3-year weekly data for pharmaceutical marketing analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from loguru import logger

from ..config import PharmaChannels, ChannelConfig


class PharmaDataGenerator:
    """
    Generates synthetic pharmaceutical marketing data with realistic patterns.

    Features:
    - 6 marketing channels with pharma-appropriate spend patterns
    - Seasonal effects (Q1 lows, Q4 highs for Rx)
    - Competitor spend as control variable
    - Pricing effects
    - Adstock/carryover effects built into the response
    """

    def __init__(
        self,
        start_date: str = "2021-01-04",
        n_weeks: int = 156,  # 3 years
        base_sales: float = 50_000_000,
        noise_level: float = 0.05,
        random_seed: Optional[int] = 42
    ):
        self.start_date = pd.to_datetime(start_date)
        self.n_weeks = n_weeks
        self.base_sales = base_sales
        self.noise_level = noise_level
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

        self.channels = PharmaChannels.get_all()
        self.true_effects: Dict[str, Dict] = {}

    def _generate_dates(self) -> pd.DatetimeIndex:
        """Generate weekly date index."""
        return pd.date_range(
            start=self.start_date,
            periods=self.n_weeks,
            freq='W-MON'
        )

    def _generate_seasonality(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Generate pharma seasonality pattern.

        - Q1: Lower (post-holiday lull, insurance resets)
        - Q2-Q3: Moderate
        - Q4: Higher (end-of-year push, holiday health concerns)
        """
        month = dates.month
        seasonality = np.ones(len(dates))

        # Q1 effect (lower)
        seasonality[(month >= 1) & (month <= 3)] = 0.9

        # Q2-Q3 effect (moderate)
        seasonality[(month >= 4) & (month <= 9)] = 1.0

        # Q4 effect (higher)
        seasonality[(month >= 10) & (month <= 12)] = 1.1

        # Add smooth sine wave for gradual transitions
        day_of_year = dates.dayofyear
        sine_seasonality = 0.05 * np.sin(2 * np.pi * (day_of_year - 90) / 365)

        return seasonality + sine_seasonality

    def _generate_trend(self, n: int) -> np.ndarray:
        """Generate underlying market trend (slight growth)."""
        # 5% annual growth with some variation
        annual_growth = 0.05
        weekly_growth = (1 + annual_growth) ** (1/52) - 1

        trend = np.cumprod(np.ones(n) * (1 + weekly_growth))
        # Add slight random walk
        random_walk = np.cumsum(np.random.normal(0, 0.002, n))

        return trend + random_walk

    def _generate_channel_spend(
        self,
        channel: ChannelConfig,
        n: int,
        dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generate realistic spend patterns for a channel."""
        min_spend, max_spend = channel.typical_spend_range
        mean_spend = (min_spend + max_spend) / 2

        # Base spend with some autocorrelation
        spend = np.zeros(n)
        spend[0] = mean_spend

        for i in range(1, n):
            # AR(1) process with reversion to mean
            spend[i] = 0.7 * spend[i-1] + 0.3 * mean_spend + np.random.normal(0, mean_spend * 0.15)

        # Add channel-specific patterns
        if channel.name == "Conferences":
            # Conferences concentrated in Q1 and Q4
            month = dates.month
            spend[(month >= 2) & (month <= 3)] *= 1.5  # ASCO, etc.
            spend[(month >= 9) & (month <= 11)] *= 1.8  # Fall conferences

        elif channel.name == "HCP Detailing":
            # Lower in summer (vacation season)
            month = dates.month
            spend[(month >= 6) & (month <= 8)] *= 0.7

        elif channel.name == "DTC TV":
            # Higher during winter months
            month = dates.month
            spend[(month >= 11) | (month <= 2)] *= 1.3

        # Ensure bounds
        spend = np.clip(spend, min_spend * 0.5, max_spend * 1.5)

        return spend

    def _apply_adstock(
        self,
        x: np.ndarray,
        decay: float,
        max_lag: int = 12
    ) -> np.ndarray:
        """Apply geometric adstock transformation."""
        adstocked = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(min(i + 1, max_lag)):
                adstocked[i] += x[i - j] * (decay ** j)
        return adstocked

    def _apply_saturation(
        self,
        x: np.ndarray,
        k: float,
        s: float
    ) -> np.ndarray:
        """
        Apply Hill saturation function.

        Parameters:
        - k: Half-saturation point (spend level at 50% effect)
        - s: Shape parameter (steepness)
        """
        # Normalize x to 0-1 range for stability
        x_norm = x / (x.max() + 1e-8)
        return x_norm ** s / (k ** s + x_norm ** s)

    def _generate_competitor_spend(
        self,
        n: int,
        dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generate competitor marketing spend (control variable)."""
        base_competitor = 30_000_000  # Monthly competitor spend

        # Weekly competitor spend with variation
        spend = base_competitor / 4 * np.ones(n)

        # Add autocorrelated noise
        for i in range(1, n):
            spend[i] = 0.8 * spend[i-1] + 0.2 * (base_competitor / 4) + \
                       np.random.normal(0, base_competitor * 0.02)

        # Competitor might increase spend in Q4
        month = dates.month
        spend[(month >= 10) & (month <= 12)] *= 1.2

        return spend

    def _generate_pricing(
        self,
        n: int,
        base_price: float = 500
    ) -> np.ndarray:
        """Generate price series (control variable)."""
        price = np.ones(n) * base_price

        # Annual price increases (typical pharma behavior)
        for i in range(n):
            year = i // 52
            price[i] = base_price * (1.05 ** year)  # 5% annual increase

        # Add small random variation
        price += np.random.normal(0, base_price * 0.01, n)

        return price

    def generate(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate complete synthetic dataset.

        Returns:
            Tuple of (DataFrame with all data, Dict with true effects)
        """
        logger.info(f"Generating {self.n_weeks} weeks of pharma marketing data...")

        dates = self._generate_dates()
        n = len(dates)

        # Initialize DataFrame
        data = pd.DataFrame({'date': dates})
        data['week_number'] = range(1, n + 1)

        # Generate base components
        seasonality = self._generate_seasonality(dates)
        trend = self._generate_trend(n)

        # Generate channel spend and calculate effects
        total_marketing_effect = np.zeros(n)

        for channel in self.channels:
            col_name = channel.name.lower().replace(' ', '_')
            spend_col = f"{col_name}_spend"

            # Generate spend
            spend = self._generate_channel_spend(channel, n, dates)
            data[spend_col] = spend

            # True effect parameters
            true_decay = np.random.uniform(
                channel.adstock_decay_min,
                channel.adstock_decay_max
            )
            true_k = np.random.uniform(0.3, 0.7)
            true_s = np.random.uniform(0.5, 1.5)
            true_beta = np.random.uniform(0.02, 0.08)

            # Store true effects
            self.true_effects[col_name] = {
                'decay': true_decay,
                'k': true_k,
                's': true_s,
                'beta': true_beta
            }

            # Apply transformations
            adstocked = self._apply_adstock(spend, true_decay)
            saturated = self._apply_saturation(adstocked, true_k, true_s)

            # Add to total effect
            total_marketing_effect += true_beta * saturated

        # Control variables
        data['competitor_spend'] = self._generate_competitor_spend(n, dates)
        data['price'] = self._generate_pricing(n)

        # Competitor effect (negative)
        competitor_effect = -0.03 * (data['competitor_spend'] / data['competitor_spend'].mean() - 1)

        # Price elasticity (negative)
        price_effect = -0.5 * (data['price'] / data['price'].mean() - 1)

        # Calculate total sales
        base_sales = self.base_sales * trend * seasonality
        marketing_multiplier = 1 + total_marketing_effect
        control_multiplier = 1 + competitor_effect + price_effect

        # Add noise
        noise = np.random.normal(1, self.noise_level, n)

        data['sales'] = base_sales * marketing_multiplier * control_multiplier * noise

        # Add some derived columns
        data['year'] = dates.year
        data['month'] = dates.month
        data['quarter'] = dates.quarter
        data['week_of_year'] = dates.isocalendar().week

        # Calculate total marketing spend
        spend_cols = [col for col in data.columns if col.endswith('_spend') and col != 'competitor_spend']
        data['total_marketing_spend'] = data[spend_cols].sum(axis=1)

        logger.info(f"Generated data with {len(data)} rows and {len(data.columns)} columns")

        return data, self.true_effects

    def save_data(
        self,
        data: pd.DataFrame,
        filepath: str = "data/sample_pharma_data.csv"
    ):
        """Save generated data to CSV."""
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")


def generate_sample_data(
    output_path: str = "data/sample_pharma_data.csv",
    n_weeks: int = 156
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to generate and save sample data.

    Args:
        output_path: Path to save CSV
        n_weeks: Number of weeks to generate

    Returns:
        Tuple of (DataFrame, true_effects dict)
    """
    generator = PharmaDataGenerator(n_weeks=n_weeks)
    data, true_effects = generator.generate()
    generator.save_data(data, output_path)
    return data, true_effects


if __name__ == "__main__":
    # Generate sample data when run directly
    data, effects = generate_sample_data()
    print(f"\nData shape: {data.shape}")
    print(f"\nColumns: {data.columns.tolist()}")
    print(f"\nSales summary:")
    print(data['sales'].describe())
    print(f"\nTrue effects: {effects}")
