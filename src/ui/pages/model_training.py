"""
Model Training Page for Pharma MMM Agent.

Handles model configuration and training.
"""

import streamlit as st
import pandas as pd
import json


def render_model_training():
    """Render the Model Training page."""
    st.header("🧮 Model Training")

    if st.session_state.get('data') is None:
        st.warning("Please load data first in the Data Overview page.")
        return

    df = st.session_state.data
    date_col = st.session_state.get('date_column', 'date')
    sales_col = st.session_state.get('sales_column', 'sales')
    spend_cols = st.session_state.get('spend_columns', [])

    st.markdown("""
    Configure and train a Bayesian Market Mix Model using PyMC.
    The model estimates the contribution of each marketing channel to sales,
    accounting for adstock (carryover) effects and saturation.
    """)

    # Model configuration
    st.subheader("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Adstock Settings")

        estimate_adstock = st.checkbox(
            "Estimate Adstock Decay",
            value=True,
            help="Let the model estimate decay rates, or use fixed values"
        )

        adstock_max_lag = st.slider(
            "Maximum Lag (weeks)",
            min_value=4,
            max_value=24,
            value=12,
            help="Maximum carryover effect duration. Pharma typically has longer carryover."
        )

        st.markdown("### Saturation Settings")

        apply_saturation = st.checkbox(
            "Apply Saturation Curves",
            value=True,
            help="Model diminishing returns using Hill function"
        )

    with col2:
        st.markdown("### MCMC Settings")

        n_samples = st.slider(
            "Number of Samples",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="More samples = more accurate but slower"
        )

        n_tune = st.slider(
            "Tuning Steps",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100
        )

        n_chains = st.selectbox(
            "Number of Chains",
            [1, 2, 4],
            index=1,
            help="More chains help assess convergence"
        )

        target_accept = st.slider(
            "Target Accept Rate",
            min_value=0.7,
            max_value=0.99,
            value=0.9,
            help="Higher values are more conservative"
        )

    # Control variables
    st.subheader("Control Variables")

    available_controls = [c for c in df.columns
                          if c not in [date_col, sales_col] + spend_cols
                          and c not in ['year', 'month', 'quarter', 'week_number', 'week_of_year', 'total_marketing_spend']]

    control_cols = st.multiselect(
        "Select Control Variables",
        available_controls,
        default=[c for c in ['competitor_spend', 'price'] if c in available_controls],
        help="Variables that affect sales but aren't marketing spend"
    )

    # Channel selection
    st.subheader("Channels to Model")

    selected_spend_cols = st.multiselect(
        "Select Marketing Channels",
        spend_cols,
        default=spend_cols,
        format_func=lambda x: x.replace('_spend', '').replace('_', ' ').title()
    )

    st.markdown("---")

    # Training section
    st.subheader("Train Model")

    # Show data summary
    st.markdown(f"""
    **Training Data Summary:**
    - Observations: {len(df)}
    - Date range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}
    - Channels: {len(selected_spend_cols)}
    - Control variables: {len(control_cols)}
    """)

    # Estimate training time
    est_time = (n_samples + n_tune) * n_chains * len(selected_spend_cols) / 1000
    st.info(f"⏱️ Estimated training time: {est_time:.0f}-{est_time*2:.0f} minutes (varies by hardware)")

    if st.button("🚀 Train Model", type="primary", disabled=len(selected_spend_cols) == 0):
        train_model(
            df, date_col, sales_col, selected_spend_cols, control_cols,
            estimate_adstock, adstock_max_lag, apply_saturation,
            n_samples, n_tune, n_chains, target_accept
        )

    # Show existing model if available
    if st.session_state.get('model_results'):
        st.markdown("---")
        st.subheader("Current Model")
        results = st.session_state.model_results

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model ID", results.model_id)
        with col2:
            st.metric("R²", f"{results.r_squared:.3f}")
        with col3:
            st.metric("MAPE", f"{results.mape:.1f}%")


def train_model(df, date_col, sales_col, spend_cols, control_cols,
                estimate_adstock, adstock_max_lag, apply_saturation,
                n_samples, n_tune, n_chains, target_accept):
    """Train the Bayesian MMM model."""
    from ...services.modeling_service import ModelingService, ModelTrainingRequest

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Preparing data...")
        progress_bar.progress(10)

        # Prepare request
        request = ModelTrainingRequest(
            data_json=df.to_json(orient='records', date_format='iso'),
            date_column=date_col,
            sales_column=sales_col,
            spend_columns=spend_cols,
            control_columns=control_cols if control_cols else None,
            adstock_max_lag=adstock_max_lag,
            estimate_adstock=estimate_adstock,
            apply_saturation=apply_saturation,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            target_accept=target_accept
        )

        status_text.text("Training model (this may take several minutes)...")
        progress_bar.progress(30)

        # Train model
        results = ModelingService.train_model(request)

        progress_bar.progress(90)

        if results.success:
            st.session_state.model_results = results
            st.session_state.model_id = results.model_id

            progress_bar.progress(100)
            status_text.text("Training complete!")

            st.success(f"Model trained successfully! ID: {results.model_id}")

            # Show quick results
            st.markdown("### Quick Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{results.r_squared:.3f}")
            with col2:
                st.metric("MAPE", f"{results.mape:.1f}%")
            with col3:
                st.metric("Base Sales %", f"{results.base_sales_percentage:.1f}%")

            # Diagnostics
            st.markdown("### Model Diagnostics")
            diag = results.diagnostics

            if diag['n_divergences'] > 0:
                st.warning(f"⚠️ {diag['n_divergences']} divergent transitions detected")
            else:
                st.success("✅ No divergent transitions")

            if diag['r_hat_max'] > 1.05:
                st.warning(f"⚠️ Max R-hat: {diag['r_hat_max']:.3f} (>1.05 suggests convergence issues)")
            else:
                st.success(f"✅ Max R-hat: {diag['r_hat_max']:.3f}")

            st.info("Go to Results Dashboard to see detailed analysis →")

        else:
            st.error("Model training failed. Please check your data and try again.")
            progress_bar.progress(0)

    except Exception as e:
        st.error(f"Training error: {str(e)}")
        progress_bar.progress(0)
        raise
