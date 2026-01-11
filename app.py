"""
Pharma Market Mix Model Agent - Streamlit Application

Main entry point for the Streamlit UI.
"""

import streamlit as st
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Pharma MMM Agent",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'model_id' not in st.session_state:
    st.session_state.model_id = None

# Sidebar navigation
st.sidebar.markdown("## 💊 Pharma MMM Agent")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data Overview", "🧮 Model Training", "📈 Results Dashboard",
     "💰 Budget Optimization", "💬 Chat Interface"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")

if st.sidebar.button("🔄 Generate Sample Data"):
    with st.spinner("Generating sample data..."):
        from src.utils.data_generator import generate_sample_data
        os.makedirs("data", exist_ok=True)
        data, _ = generate_sample_data("data/sample_pharma_data.csv")
        st.session_state.data = data
        st.sidebar.success("Sample data generated!")

if st.sidebar.button("🗑️ Clear Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main content based on page selection
if page == "🏠 Home":
    st.markdown('<p class="main-header">Pharma Market Mix Model Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered marketing analytics for pharmaceutical companies</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Data Analysis")
        st.markdown("""
        - Upload marketing spend data
        - Validate data quality
        - Explore channel metrics
        - Generate sample data
        """)

    with col2:
        st.markdown("### 🧮 Bayesian MMM")
        st.markdown("""
        - PyMC-based modeling
        - Adstock transformations
        - Saturation curves
        - Channel decomposition
        """)

    with col3:
        st.markdown("### 💡 AI Insights")
        st.markdown("""
        - Claude-powered Q&A
        - Budget optimization
        - Natural language summaries
        - Actionable recommendations
        """)

    st.markdown("---")

    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Generate or Upload Data**: Use the sidebar to generate sample data or upload your own
    2. **Explore Data**: Review data quality and channel spend patterns
    3. **Train Model**: Configure and train the Bayesian MMM
    4. **Analyze Results**: View channel contributions and ROI
    5. **Optimize Budget**: Get AI-powered budget recommendations
    6. **Chat**: Ask questions about your results in natural language
    """)

    st.markdown("---")

    st.markdown("### 📋 Pharma-Specific Features")

    features_col1, features_col2 = st.columns(2)

    with features_col1:
        st.markdown("""
        **Marketing Channels:**
        - DTC TV & Digital
        - HCP Detailing
        - Medical Conferences
        - Samples Distribution
        - Journal Advertising
        """)

    with features_col2:
        st.markdown("""
        **Modeling Features:**
        - Longer carryover (4-12 weeks)
        - Weibull adstock option
        - Competitor spend controls
        - Seasonality adjustment
        - Price elasticity
        """)

elif page == "📊 Data Overview":
    from src.ui.pages.data_overview import render_data_overview
    render_data_overview()

elif page == "🧮 Model Training":
    from src.ui.pages.model_training import render_model_training
    render_model_training()

elif page == "📈 Results Dashboard":
    from src.ui.pages.results_dashboard import render_results_dashboard
    render_results_dashboard()

elif page == "💰 Budget Optimization":
    from src.ui.pages.budget_optimization import render_budget_optimization
    render_budget_optimization()

elif page == "💬 Chat Interface":
    from src.ui.pages.chat_interface import render_chat_interface
    render_chat_interface()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
Built for n8n workflow integration.
API endpoints available at `/api/v1/`
""")
st.sidebar.markdown("v0.1.0")
