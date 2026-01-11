"""
Budget Optimization Page for Pharma MMM Agent.

Handles budget allocation optimization and scenario analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json


def render_budget_optimization():
    """Render the Budget Optimization page."""
    st.header("💰 Budget Optimization")

    if st.session_state.get('model_results') is None:
        st.warning("No model results available. Please train a model first.")
        return

    results = st.session_state.model_results
    contributions = results.channel_contributions

    st.markdown("""
    Use the trained model to optimize budget allocation across channels.
    The optimizer finds the allocation that maximizes expected sales given constraints.
    """)

    # Current allocation
    st.subheader("Current Budget Allocation")

    current_allocation = {c.channel: c.total_spend for c in contributions}
    total_current = sum(current_allocation.values())

    # Display current allocation
    col1, col2 = st.columns([2, 1])

    with col1:
        current_df = pd.DataFrame([
            {'Channel': k, 'Current Spend': v, 'Share': v/total_current*100}
            for k, v in current_allocation.items()
        ])

        fig_current = px.pie(
            current_df,
            values='Current Spend',
            names='Channel',
            title="Current Budget Allocation"
        )
        st.plotly_chart(fig_current, use_container_width=True)

    with col2:
        st.metric("Total Budget", f"${total_current:,.0f}")
        st.dataframe(
            current_df.style.format({
                'Current Spend': '${:,.0f}',
                'Share': '{:.1f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Optimization settings
    st.subheader("Optimization Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_budget = st.number_input(
            "Total Budget ($)",
            min_value=0,
            value=int(total_current),
            step=100000,
            help="Total budget to allocate across channels"
        )

    with col2:
        optimization_objective = st.selectbox(
            "Optimization Objective",
            ["maximize_sales", "maximize_roi"],
            format_func=lambda x: "Maximize Sales" if x == "maximize_sales" else "Maximize ROI"
        )

    with col3:
        n_periods = st.number_input(
            "Optimization Period (weeks)",
            min_value=1,
            max_value=104,
            value=52,
            help="Number of weeks to optimize for"
        )

    # Channel constraints
    st.subheader("Channel Constraints")

    st.markdown("Set minimum and maximum spend limits for each channel.")

    constraints_data = []

    for channel in current_allocation.keys():
        current = current_allocation[channel]

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{channel}**")
            st.caption(f"Current: ${current:,.0f}")

        with col2:
            min_ratio = st.slider(
                f"Min % ({channel})",
                min_value=0,
                max_value=100,
                value=50,
                key=f"min_{channel}",
                help="Minimum spend as % of current"
            )

        with col3:
            max_ratio = st.slider(
                f"Max % ({channel})",
                min_value=100,
                max_value=300,
                value=200,
                key=f"max_{channel}",
                help="Maximum spend as % of current"
            )

        constraints_data.append({
            'channel': channel,
            'min_ratio': min_ratio / 100,
            'max_ratio': max_ratio / 100
        })

    st.markdown("---")

    # Run optimization
    if st.button("🚀 Optimize Budget", type="primary"):
        run_optimization(
            results, current_allocation, total_budget,
            optimization_objective, n_periods, constraints_data
        )

    # Show optimization results if available
    if st.session_state.get('optimization_results'):
        render_optimization_results(current_allocation)


def run_optimization(results, current_allocation, total_budget,
                     objective, n_periods, constraints):
    """Run budget optimization."""
    from ...services.optimization_service import (
        OptimizationService, OptimizationRequest, ChannelConstraint
    )

    with st.spinner("Optimizing budget allocation..."):
        try:
            # Build request
            channel_constraints = [
                ChannelConstraint(
                    channel=c['channel'],
                    min_ratio=c['min_ratio'],
                    max_ratio=c['max_ratio']
                )
                for c in constraints
            ]

            request = OptimizationRequest(
                model_id=results.model_id,
                total_budget=total_budget,
                current_allocation=current_allocation,
                constraints=channel_constraints,
                optimization_objective=objective,
                n_periods=n_periods
            )

            opt_results = OptimizationService.optimize_budget(request)

            if opt_results.success:
                st.session_state.optimization_results = opt_results
                st.success("Optimization complete!")
            else:
                st.error(f"Optimization failed: {opt_results.message}")

        except Exception as e:
            st.error(f"Optimization error: {str(e)}")


def render_optimization_results(current_allocation):
    """Render optimization results."""
    opt_results = st.session_state.optimization_results

    st.markdown("---")
    st.subheader("Optimization Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Expected Sales",
            f"${opt_results.current_expected_sales:,.0f}"
        )

    with col2:
        st.metric(
            "Optimized Expected Sales",
            f"${opt_results.optimized_expected_sales:,.0f}"
        )

    with col3:
        st.metric(
            "Improvement",
            f"{opt_results.improvement_percentage:+.1f}%",
            delta=f"${opt_results.optimized_expected_sales - opt_results.current_expected_sales:,.0f}"
        )

    with col4:
        st.metric("Total Budget", f"${opt_results.total_budget:,.0f}")

    # Allocation comparison
    st.markdown("### Budget Reallocation")

    alloc_data = []
    for alloc in opt_results.allocations:
        alloc_data.append({
            'Channel': alloc.channel,
            'Current ($)': alloc.current_spend,
            'Optimized ($)': alloc.optimized_spend,
            'Change (%)': alloc.change_percentage,
            'Marginal ROI': alloc.marginal_roi
        })

    alloc_df = pd.DataFrame(alloc_data)

    # Bar chart comparison
    fig_compare = go.Figure()

    fig_compare.add_trace(go.Bar(
        name='Current',
        x=alloc_df['Channel'],
        y=alloc_df['Current ($)'],
        marker_color='lightblue'
    ))

    fig_compare.add_trace(go.Bar(
        name='Optimized',
        x=alloc_df['Channel'],
        y=alloc_df['Optimized ($)'],
        marker_color='darkblue'
    ))

    fig_compare.update_layout(
        title="Current vs Optimized Budget Allocation",
        barmode='group',
        height=400
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Change visualization
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Recommended Changes")

        # Sort by change magnitude
        sorted_alloc = sorted(alloc_data, key=lambda x: x['Change (%)'], reverse=True)

        for item in sorted_alloc:
            change = item['Change (%)']
            if change > 0:
                st.markdown(f"📈 **{item['Channel']}**: Increase by {change:.1f}%")
            elif change < 0:
                st.markdown(f"📉 **{item['Channel']}**: Decrease by {abs(change):.1f}%")
            else:
                st.markdown(f"➡️ **{item['Channel']}**: No change recommended")

    with col2:
        st.markdown("### Marginal ROI at Optimized Levels")

        fig_mroi = px.bar(
            alloc_df.sort_values('Marginal ROI', ascending=True),
            x='Marginal ROI',
            y='Channel',
            orientation='h',
            title="Marginal ROI",
            color='Marginal ROI',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_mroi, use_container_width=True)

    # Detailed table
    st.markdown("### Detailed Allocation Table")

    st.dataframe(
        alloc_df.style.format({
            'Current ($)': '${:,.0f}',
            'Optimized ($)': '${:,.0f}',
            'Change (%)': '{:+.1f}%',
            'Marginal ROI': '{:.3f}'
        }).background_gradient(subset=['Change (%)'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True
    )

    # Scenario comparison
    st.markdown("---")
    st.subheader("Scenario Analysis")

    st.markdown("Compare different budget scenarios.")

    scenario_tab1, scenario_tab2 = st.tabs(["Quick Scenarios", "Custom Scenario"])

    with scenario_tab1:
        render_quick_scenarios(opt_results)

    with scenario_tab2:
        render_custom_scenario(opt_results)


def render_quick_scenarios(opt_results):
    """Render quick scenario comparisons."""
    scenarios = {
        "Current": {alloc.channel: alloc.current_spend for alloc in opt_results.allocations},
        "Optimized": {alloc.channel: alloc.optimized_spend for alloc in opt_results.allocations},
        "+20% Budget": {alloc.channel: alloc.optimized_spend * 1.2 for alloc in opt_results.allocations},
        "-20% Budget": {alloc.channel: alloc.optimized_spend * 0.8 for alloc in opt_results.allocations}
    }

    scenario_data = []
    for name, allocation in scenarios.items():
        total = sum(allocation.values())
        scenario_data.append({
            'Scenario': name,
            'Total Budget': total,
            'Total': f"${total:,.0f}"
        })

    st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, hide_index=True)


def render_custom_scenario(opt_results):
    """Render custom scenario builder."""
    st.markdown("Build a custom scenario by adjusting channel spends.")

    custom_allocation = {}

    for alloc in opt_results.allocations:
        custom_allocation[alloc.channel] = st.number_input(
            f"{alloc.channel}",
            min_value=0,
            value=int(alloc.optimized_spend),
            step=10000,
            key=f"custom_{alloc.channel}"
        )

    total_custom = sum(custom_allocation.values())
    st.metric("Total Custom Budget", f"${total_custom:,.0f}")

    if st.button("Evaluate Custom Scenario"):
        st.info("Custom scenario evaluation would use the ScenarioAnalysis service.")
