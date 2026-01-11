"""Tests for DataService."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.data_service import (
    DataService, DataIngestionRequest, DataValidationResult,
    AdstockTransformRequest
)
from src.utils.data_generator import PharmaDataGenerator


class TestDataGenerator:
    """Tests for the data generator."""

    def test_generator_creates_data(self):
        """Test that generator creates data with expected shape."""
        generator = PharmaDataGenerator(n_weeks=52, random_seed=42)
        data, effects = generator.generate()

        assert len(data) == 52
        assert 'date' in data.columns
        assert 'sales' in data.columns

    def test_generator_creates_spend_columns(self):
        """Test that all spend columns are created."""
        generator = PharmaDataGenerator(n_weeks=52, random_seed=42)
        data, effects = generator.generate()

        expected_spend_cols = [
            'dtc_tv_spend', 'dtc_digital_spend', 'hcp_detailing_spend',
            'conferences_spend', 'samples_spend', 'journal_ads_spend'
        ]

        for col in expected_spend_cols:
            assert col in data.columns

    def test_generator_effects_match_channels(self):
        """Test that true effects are generated for each channel."""
        generator = PharmaDataGenerator(n_weeks=52, random_seed=42)
        data, effects = generator.generate()

        assert len(effects) == 6  # 6 channels
        for channel, params in effects.items():
            assert 'decay' in params
            assert 'k' in params
            assert 's' in params
            assert 'beta' in params


class TestDataValidation:
    """Tests for data validation."""

    def test_valid_data_passes(self):
        """Test that valid data passes validation."""
        generator = PharmaDataGenerator(n_weeks=52, random_seed=42)
        data, _ = generator.generate()

        validation = DataService.validate_data(
            data,
            date_column='date',
            sales_column='sales'
        )

        assert validation.is_valid
        assert validation.row_count == 52
        assert len(validation.errors) == 0

    def test_missing_date_column_fails(self):
        """Test that missing date column fails validation."""
        data = pd.DataFrame({'sales': [1, 2, 3]})

        validation = DataService.validate_data(
            data,
            date_column='date',
            sales_column='sales'
        )

        assert not validation.is_valid
        assert any('date' in e.lower() for e in validation.errors)

    def test_missing_sales_column_fails(self):
        """Test that missing sales column fails validation."""
        data = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=10, freq='W')
        })

        validation = DataService.validate_data(
            data,
            date_column='date',
            sales_column='sales'
        )

        assert not validation.is_valid
        assert any('sales' in e.lower() for e in validation.errors)

    def test_empty_dataframe_fails(self):
        """Test that empty DataFrame fails validation."""
        data = pd.DataFrame()

        validation = DataService.validate_data(data)

        assert not validation.is_valid

    def test_negative_sales_fails(self):
        """Test that negative sales values fail validation."""
        data = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=10, freq='W'),
            'sales': [-100, 200, 300, 400, 500, -600, 700, 800, 900, 1000]
        })

        validation = DataService.validate_data(
            data,
            date_column='date',
            sales_column='sales'
        )

        assert not validation.is_valid
        assert any('negative' in e.lower() for e in validation.errors)


class TestDataSummary:
    """Tests for data summary generation."""

    def test_summary_includes_channels(self):
        """Test that summary includes channel information."""
        generator = PharmaDataGenerator(n_weeks=52, random_seed=42)
        data, _ = generator.generate()

        summary = DataService.get_data_summary(data)

        assert len(summary.channels) == 6
        assert summary.row_count == 52
        assert summary.total_sales > 0

    def test_summary_spend_stats(self):
        """Test that spend statistics are calculated correctly."""
        generator = PharmaDataGenerator(n_weeks=52, random_seed=42)
        data, _ = generator.generate()

        summary = DataService.get_data_summary(data)

        for channel, stats in summary.channel_spend_summary.items():
            assert 'total' in stats
            assert 'mean' in stats
            assert 'std' in stats
            assert stats['total'] > 0


class TestAdstockTransform:
    """Tests for adstock transformation."""

    def test_geometric_adstock(self):
        """Test geometric adstock transformation."""
        data = pd.DataFrame({
            'spend': [100, 0, 0, 0, 0]
        })

        request = AdstockTransformRequest(
            data_json=data.to_json(),
            spend_column='spend',
            decay_rate=0.5,
            max_lag=4,
            adstock_type='geometric'
        )

        result = DataService.apply_adstock_transform(request)
        result_df = pd.read_json(result.data_json)

        # First value should be 100
        assert result_df['spend_adstock'].iloc[0] == 100

        # Second value should be 100 * 0.5 = 50
        assert result_df['spend_adstock'].iloc[1] == 50

        # Third value should be 100 * 0.5^2 = 25
        assert result_df['spend_adstock'].iloc[2] == 25

    def test_adstock_decay_bounds(self):
        """Test that decay rate is enforced."""
        data = pd.DataFrame({'spend': [100, 100, 100]})

        # Test invalid decay rate
        with pytest.raises(ValueError):
            AdstockTransformRequest(
                data_json=data.to_json(),
                spend_column='spend',
                decay_rate=1.5,  # Invalid: > 1
                adstock_type='geometric'
            )


class TestPreprocessing:
    """Tests for data preprocessing."""

    def test_preprocessing_handles_missing(self):
        """Test that preprocessing handles missing values."""
        data = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=10, freq='W'),
            'sales': [100, np.nan, 300, 400, 500, 600, 700, 800, 900, 1000],
            'spend_spend': [50, 60, np.nan, 80, 90, 100, 110, 120, 130, 140]
        })

        preprocessed = DataService.preprocess_data(
            data,
            date_column='date',
            sales_column='sales',
            spend_columns=['spend_spend']
        )

        result_df = pd.read_json(preprocessed.data_json)

        # Check no missing values
        assert result_df['sales'].isna().sum() == 0
        assert result_df['spend_spend'].isna().sum() == 0

    def test_preprocessing_sorts_by_date(self):
        """Test that preprocessing sorts data by date."""
        dates = pd.date_range('2021-01-01', periods=5, freq='W')
        data = pd.DataFrame({
            'date': dates[[4, 2, 0, 3, 1]],  # Scrambled order
            'sales': [500, 300, 100, 400, 200]
        })

        preprocessed = DataService.preprocess_data(
            data,
            date_column='date',
            sales_column='sales'
        )

        result_df = pd.read_json(preprocessed.data_json)
        result_df['date'] = pd.to_datetime(result_df['date'])

        # Check dates are sorted
        assert result_df['date'].is_monotonic_increasing


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
