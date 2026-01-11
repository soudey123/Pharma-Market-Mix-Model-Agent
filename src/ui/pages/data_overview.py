"""
Data Overview Page for Pharma MMM Agent.

Handles data upload, validation, and exploration.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from ...services.data_service import DataService, DataIngestionRequest


def render_data_overview():
    """Render the Data Overview page."""
    st.header("📊 Data Overview")

    # Data source selection
    tab1, tab2 = st.tabs(["📁 Upload Data", "📋 Current Data"])

    with tab1:
        render_upload_section()

    with tab2:
        render_data_exploration()


def render_upload_section():
    """Render the data upload section."""
    st.subheader("Upload Your Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx"],
            help="Upload weekly marketing data with spend columns and sales"
        )

        if uploaded_file:
            # Save temporarily
            temp_path = f"data/temp_{uploaded_file.name}"
            os.makedirs("data", exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Configure columns
            st.markdown("### Configure Columns")

            preview_df = pd.read_csv(temp_path) if temp_path.endswith('.csv') else pd.read_excel(temp_path)
            columns = preview_df.columns.tolist()

            date_col = st.selectbox("Date Column", columns, index=0 if columns else None)
            sales_col = st.selectbox("Sales/Response Column",
                                     columns,
                                     index=columns.index('sales') if 'sales' in columns else 0)

            # Auto-detect spend columns
            default_spend = [c for c in columns if c.endswith('_spend')]
            spend_cols = st.multiselect(
                "Spend Columns",
                [c for c in columns if c not in [date_col, sales_col]],
                default=default_spend
            )

            if st.button("✅ Load Data", type="primary"):
                with st.spinner("Loading and validating data..."):
                    request = DataIngestionRequest(
                        file_path=temp_path,
                        date_column=date_col,
                        sales_column=sales_col,
                        spend_columns=spend_cols
                    )

                    df, validation = DataService.ingest_data(request)

                    if validation.is_valid:
                        st.session_state.data = df
                        st.session_state.date_column = date_col
                        st.session_state.sales_column = sales_col
                        st.session_state.spend_columns = spend_cols
                        st.success(f"Data loaded successfully! {validation.row_count} rows, {validation.column_count} columns")

                        # Show validation summary
                        if validation.warnings:
                            for warning in validation.warnings:
                                st.warning(warning)
                    else:
                        for error in validation.errors:
                            st.error(error)

    with col2:
        st.markdown("### Expected Format")
        st.markdown("""
        Your data should include:
        - **Date column**: Weekly dates
        - **Sales column**: Response variable
        - **Spend columns**: Marketing spend by channel

        Optional:
        - Competitor spend
        - Price
        - Other control variables
        """)

        st.markdown("### Sample Data")
        if st.button("📥 Load Sample Data"):
            sample_path = "data/sample_pharma_data.csv"
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                df['date'] = pd.to_datetime(df['date'])
                st.session_state.data = df
                st.session_state.date_column = 'date'
                st.session_state.sales_column = 'sales'
                st.session_state.spend_columns = [c for c in df.columns if c.endswith('_spend') and c != 'competitor_spend']
                st.success("Sample data loaded!")
            else:
                st.warning("Generate sample data first using the sidebar button")


def render_data_exploration():
    """Render data exploration section."""
    if st.session_state.get('data') is None:
        st.info("No data loaded. Upload data or generate sample data to begin.")
        return

    df = st.session_state.data
    date_col = st.session_state.get('date_column', 'date')
    sales_col = st.session_state.get('sales_column', 'sales')
    spend_cols = st.session_state.get('spend_columns', [])

    # Summary metrics
    st.subheader("Data Summary")

    summary = DataService.get_data_summary(df, date_col, sales_col)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{summary.row_count:,}")
    with col2:
        st.metric("Date Range", f"{summary.date_range['start']} to {summary.date_range['end']}")
    with col3:
        st.metric("Total Sales", f"${summary.total_sales:,.0f}")
    with col4:
        st.metric("Avg Weekly Sales", f"${summary.avg_weekly_sales:,.0f}")

    st.markdown("---")

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    # Visualizations
    st.subheader("Data Visualization")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Time Series", "📊 Spend Distribution", "🔗 Correlations"])

    with viz_tab1:
        render_time_series(df, date_col, sales_col, spend_cols)

    with viz_tab2:
        render_spend_distribution(df, spend_cols)

    with viz_tab3:
        render_correlations(df, sales_col, spend_cols)


def render_time_series(df, date_col, sales_col, spend_cols):
    """Render time series visualizations."""
    # Sales over time
    fig_sales = px.line(
        df, x=date_col, y=sales_col,
        title="Sales Over Time",
        labels={date_col: "Date", sales_col: "Sales ($)"}
    )
    fig_sales.update_layout(height=400)
    st.plotly_chart(fig_sales, use_container_width=True)

    # Marketing spend over time
    if spend_cols:
        spend_data = df[[date_col] + spend_cols].melt(
            id_vars=[date_col],
            value_vars=spend_cols,
            var_name='Channel',
            value_name='Spend'
        )
        spend_data['Channel'] = spend_data['Channel'].str.replace('_spend', '').str.replace('_', ' ').str.title()

        fig_spend = px.area(
            spend_data, x=date_col, y='Spend', color='Channel',
            title="Marketing Spend by Channel Over Time"
        )
        fig_spend.update_layout(height=400)
        st.plotly_chart(fig_spend, use_container_width=True)


def render_spend_distribution(df, spend_cols):
    """Render spend distribution visualizations."""
    if not spend_cols:
        st.info("No spend columns detected")
        return

    # Total spend by channel
    total_spend = {col.replace('_spend', '').replace('_', ' ').title(): df[col].sum()
                   for col in spend_cols}

    fig_pie = px.pie(
        values=list(total_spend.values()),
        names=list(total_spend.keys()),
        title="Total Spend by Channel"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Box plots
    spend_data = df[spend_cols].melt(var_name='Channel', value_name='Spend')
    spend_data['Channel'] = spend_data['Channel'].str.replace('_spend', '').str.replace('_', ' ').str.title()

    fig_box = px.box(
        spend_data, x='Channel', y='Spend',
        title="Spend Distribution by Channel"
    )
    fig_box.update_layout(height=400)
    st.plotly_chart(fig_box, use_container_width=True)


def render_correlations(df, sales_col, spend_cols):
    """Render correlation analysis."""
    if not spend_cols:
        st.info("No spend columns detected")
        return

    # Correlation matrix
    corr_cols = [sales_col] + spend_cols
    corr_matrix = df[corr_cols].corr()

    # Rename for display
    display_names = {sales_col: 'Sales'}
    display_names.update({col: col.replace('_spend', '').replace('_', ' ').title()
                          for col in spend_cols})

    corr_matrix = corr_matrix.rename(index=display_names, columns=display_names)

    fig_corr = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=".2f"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter plots
    st.markdown("### Sales vs Spend")

    selected_channel = st.selectbox(
        "Select Channel",
        spend_cols,
        format_func=lambda x: x.replace('_spend', '').replace('_', ' ').title()
    )

    fig_scatter = px.scatter(
        df, x=selected_channel, y=sales_col,
        trendline="ols",
        title=f"Sales vs {selected_channel.replace('_spend', '').replace('_', ' ').title()} Spend",
        labels={selected_channel: "Spend ($)", sales_col: "Sales ($)"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
