"""
Results Dashboard Page for Pharma MMM Agent.

Displays model results with visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np


def render_results_dashboard():
    """Render the Results Dashboard page."""
    st.header("📈 Results Dashboard")

    if st.session_state.get('model_results') is None:
        st.warning("No model results available. Please train a model first.")
        return

    results = st.session_state.model_results

    # Overview metrics
    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model R²", f"{results.r_squared:.3f}",
                  help="Proportion of variance explained")
    with col2:
        st.metric("MAPE", f"{results.mape:.1f}%",
                  help="Mean Absolute Percentage Error")
    with col3:
        st.metric("Base Sales", f"${results.base_sales:,.0f}",
                  help="Sales not attributed to marketing")
    with col4:
        st.metric("Marketing Impact", f"${results.total_marketing_contribution:,.0f}",
                  help="Total sales from marketing")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Channel Contributions",
        "📈 Response Curves",
        "🔍 Model Fit",
        "📋 Coefficients"
    ])

    with tab1:
        render_channel_contributions(results)

    with tab2:
        render_response_curves(results)

    with tab3:
        render_model_fit(results)

    with tab4:
        render_coefficients(results)


def render_channel_contributions(results):
    """Render channel contribution visualizations."""
    st.subheader("Channel Contributions")

    contributions = results.channel_contributions

    # Prepare data
    contrib_data = pd.DataFrame([
        {
            'Channel': c.channel,
            'Contribution ($)': c.total_contribution,
            'Contribution (%)': c.contribution_percentage,
            'ROI': c.roi,
            'Total Spend ($)': c.total_spend
        }
        for c in contributions
    ])

    # Waterfall chart
    st.markdown("### Contribution Waterfall")

    waterfall_data = [
        {'name': 'Base Sales', 'value': results.base_sales, 'type': 'base'}
    ]

    for c in contributions:
        waterfall_data.append({
            'name': c.channel,
            'value': c.total_contribution,
            'type': 'channel'
        })

    waterfall_data.append({
        'name': 'Total Sales',
        'value': results.base_sales + results.total_marketing_contribution,
        'type': 'total'
    })

    fig_waterfall = go.Figure(go.Waterfall(
        x=[d['name'] for d in waterfall_data],
        y=[d['value'] for d in waterfall_data],
        measure=['absolute'] + ['relative'] * len(contributions) + ['total'],
        textposition='outside',
        text=[f"${d['value']:,.0f}" for d in waterfall_data],
        connector={"line": {"color": "rgb(63, 63, 63)"}}
    ))

    fig_waterfall.update_layout(
        title="Sales Decomposition",
        showlegend=False,
        height=500
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Contribution pie chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Contribution Share")

        pie_data = pd.DataFrame([
            {'Component': 'Base Sales', 'Value': results.base_sales_percentage}
        ] + [
            {'Component': c.channel, 'Value': c.contribution_percentage}
            for c in contributions
        ])

        fig_pie = px.pie(
            pie_data,
            values='Value',
            names='Component',
            title="Sales Attribution (%)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### ROI by Channel")

        fig_roi = px.bar(
            contrib_data.sort_values('ROI', ascending=True),
            x='ROI',
            y='Channel',
            orientation='h',
            title="Return on Investment",
            color='ROI',
            color_continuous_scale='RdYlGn'
        )
        fig_roi.update_layout(height=400)
        st.plotly_chart(fig_roi, use_container_width=True)

    # Channel metrics table
    st.markdown("### Channel Performance Summary")
    st.dataframe(
        contrib_data.style.format({
            'Contribution ($)': '${:,.0f}',
            'Contribution (%)': '{:.1f}%',
            'ROI': '{:.2f}',
            'Total Spend ($)': '${:,.0f}'
        }),
        use_container_width=True
    )


def render_response_curves(results):
    """Render response curve visualizations."""
    st.subheader("Response Curves")

    st.markdown("""
    Response curves show the relationship between spend and incremental sales,
    accounting for saturation (diminishing returns) effects.
    """)

    contributions = results.channel_contributions

    # Generate response curves
    for contrib in contributions:
        if contrib.saturation_k and contrib.saturation_s:
            col1, col2 = st.columns([2, 1])

            with col1:
                # Generate curve
                spend_range = np.linspace(0, contrib.total_spend * 2, 100)
                spend_normalized = spend_range / (spend_range.max() + 1e-8)

                k = contrib.saturation_k
                s = contrib.saturation_s

                response = spend_normalized ** s / (k ** s + spend_normalized ** s)
                response_scaled = response * contrib.coefficient * contrib.total_spend

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=spend_range,
                    y=response_scaled,
                    mode='lines',
                    name='Response Curve',
                    line=dict(color='blue', width=2)
                ))

                # Add current spend marker
                current_response = (contrib.total_spend / (contrib.total_spend * 2)) ** s / (
                    k ** s + (contrib.total_spend / (contrib.total_spend * 2)) ** s
                )
                current_response_scaled = current_response * contrib.coefficient * contrib.total_spend

                fig.add_trace(go.Scatter(
                    x=[contrib.total_spend],
                    y=[current_response_scaled],
                    mode='markers',
                    name='Current Spend',
                    marker=dict(size=15, color='red')
                ))

                fig.update_layout(
                    title=f"{contrib.channel} Response Curve",
                    xaxis_title="Spend ($)",
                    yaxis_title="Incremental Sales ($)",
                    height=350
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(f"**{contrib.channel}**")
                st.markdown(f"- Adstock Decay: {contrib.adstock_decay:.2f}")
                st.markdown(f"- Saturation K: {contrib.saturation_k:.2f}")
                st.markdown(f"- Saturation S: {contrib.saturation_s:.2f}")
                st.markdown(f"- Current ROI: {contrib.roi:.2f}")

                # Efficiency indicator
                if contrib.roi > 1.5:
                    st.success("High efficiency - consider increasing spend")
                elif contrib.roi < 0.5:
                    st.warning("Low efficiency - consider reducing spend")
                else:
                    st.info("Moderate efficiency")


def render_model_fit(results):
    """Render model fit visualizations."""
    st.subheader("Model Fit Analysis")

    # Get fitted values and residuals
    fitted = json.loads(results.fitted_values_json)
    residuals = json.loads(results.residuals_json)

    if st.session_state.get('data') is not None:
        df = st.session_state.data
        date_col = st.session_state.get('date_column', 'date')
        sales_col = st.session_state.get('sales_column', 'sales')

        # Actual vs Fitted
        st.markdown("### Actual vs Fitted Values")

        fig_fit = go.Figure()

        fig_fit.add_trace(go.Scatter(
            x=df[date_col],
            y=df[sales_col],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))

        fig_fit.add_trace(go.Scatter(
            x=df[date_col],
            y=fitted,
            mode='lines',
            name='Fitted',
            line=dict(color='red', dash='dash')
        ))

        fig_fit.update_layout(
            title="Actual vs Fitted Sales",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            height=400
        )

        st.plotly_chart(fig_fit, use_container_width=True)

        # Residual analysis
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Residuals Over Time")

            fig_resid_time = px.scatter(
                x=df[date_col],
                y=residuals,
                labels={'x': 'Date', 'y': 'Residual'},
                title="Residuals Over Time"
            )
            fig_resid_time.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_resid_time, use_container_width=True)

        with col2:
            st.markdown("### Residual Distribution")

            fig_resid_hist = px.histogram(
                x=residuals,
                nbins=30,
                labels={'x': 'Residual', 'y': 'Count'},
                title="Residual Distribution"
            )
            st.plotly_chart(fig_resid_hist, use_container_width=True)

        # Scatter plot
        st.markdown("### Actual vs Predicted")

        fig_scatter = px.scatter(
            x=fitted,
            y=df[sales_col],
            labels={'x': 'Predicted Sales', 'y': 'Actual Sales'},
            title="Actual vs Predicted"
        )

        # Add 45-degree line
        min_val = min(min(fitted), df[sales_col].min())
        max_val = max(max(fitted), df[sales_col].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ))

        st.plotly_chart(fig_scatter, use_container_width=True)


def render_coefficients(results):
    """Render coefficient estimates."""
    st.subheader("Model Coefficients")

    # Channel coefficients
    st.markdown("### Channel Coefficients (with 95% Credible Intervals)")

    coef_data = []
    for name, values in results.coefficients.items():
        coef_data.append({
            'Channel': name.replace('_spend', '').replace('_', ' ').title(),
            'Mean': values['mean'],
            'Std': values['std'],
            'Lower (2.5%)': values['lower'],
            'Upper (97.5%)': values['upper']
        })

    coef_df = pd.DataFrame(coef_data)

    # Forest plot
    fig_forest = go.Figure()

    for i, row in coef_df.iterrows():
        fig_forest.add_trace(go.Scatter(
            x=[row['Lower (2.5%)'], row['Upper (97.5%)']],
            y=[row['Channel'], row['Channel']],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))

        fig_forest.add_trace(go.Scatter(
            x=[row['Mean']],
            y=[row['Channel']],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name=row['Channel'],
            showlegend=False
        ))

    fig_forest.add_vline(x=0, line_dash="dash", line_color="red")

    fig_forest.update_layout(
        title="Coefficient Estimates (95% CI)",
        xaxis_title="Effect Size",
        height=400
    )

    st.plotly_chart(fig_forest, use_container_width=True)

    # Table
    st.markdown("### Coefficient Summary")
    st.dataframe(
        coef_df.style.format({
            'Mean': '{:.4f}',
            'Std': '{:.4f}',
            'Lower (2.5%)': '{:.4f}',
            'Upper (97.5%)': '{:.4f}'
        }),
        use_container_width=True
    )

    # Model diagnostics
    st.markdown("### Model Diagnostics")

    diag = results.diagnostics

    diag_df = pd.DataFrame([
        {'Metric': 'Divergent Transitions', 'Value': diag['n_divergences'],
         'Status': '✅ Good' if diag['n_divergences'] == 0 else '⚠️ Check'},
        {'Metric': 'Max R-hat', 'Value': f"{diag['r_hat_max']:.3f}",
         'Status': '✅ Good' if diag['r_hat_max'] < 1.05 else '⚠️ Check'},
        {'Metric': 'Min ESS', 'Value': f"{diag['ess_min']:.0f}",
         'Status': '✅ Good' if diag['ess_min'] > 400 else '⚠️ Check'}
    ])

    st.dataframe(diag_df, use_container_width=True, hide_index=True)
