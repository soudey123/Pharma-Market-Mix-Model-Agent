"""
Chat Interface Page for Pharma MMM Agent.

Natural language Q&A about model results using Claude API.
"""

import streamlit as st
import json
import os


def render_chat_interface():
    """Render the Chat Interface page."""
    st.header("💬 Chat Interface")

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        st.warning("""
        **Anthropic API Key Required**

        To use the chat interface, please set your API key:

        1. Create a `.env` file in the project root
        2. Add: `ANTHROPIC_API_KEY=your-api-key`
        3. Restart the application

        Or set it in your environment variables.
        """)

        # Allow manual entry for demo
        api_key = st.text_input("Or enter API key here:", type="password")

        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            st.success("API key set for this session!")
            st.rerun()

        return

    if st.session_state.get('model_results') is None:
        st.info("""
        No model results available yet.

        You can still ask general questions about Market Mix Modeling,
        but for specific insights, please train a model first.
        """)
        has_model = False
    else:
        has_model = True
        st.success("Model results loaded. Ask questions about your analysis!")

    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    # Quick actions
    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Generate Summary", disabled=not has_model):
            generate_summary()

    with col2:
        if st.button("💡 Get Recommendations", disabled=not has_model):
            generate_recommendations()

    with col3:
        if st.button("📋 Create Report", disabled=not has_model):
            generate_report()

    st.markdown("---")

    # Chat interface
    st.subheader("Ask Questions")

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your marketing analysis..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chat_response(prompt, has_model)
                st.markdown(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    # Suggested questions
    st.markdown("---")
    st.subheader("Suggested Questions")

    suggestions = [
        "Which channel has the highest ROI?",
        "What is the optimal budget allocation?",
        "How does adstock affect HCP detailing?",
        "What are the main drivers of sales?",
        "Should I increase or decrease DTC TV spend?",
        "Explain the saturation effect for digital advertising"
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggest_{i}"):
                st.session_state.chat_messages.append({"role": "user", "content": suggestion})
                st.rerun()

    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()


def generate_chat_response(question: str, has_model: bool) -> str:
    """Generate a response to a user question."""
    from ...services.insight_service import InsightService, QARequest

    try:
        service = InsightService()

        if has_model:
            results = st.session_state.model_results
            model_json = json.dumps({
                'r_squared': results.r_squared,
                'mape': results.mape,
                'base_sales': results.base_sales,
                'base_sales_percentage': results.base_sales_percentage,
                'total_marketing_contribution': results.total_marketing_contribution,
                'channel_contributions': [
                    {
                        'channel': c.channel,
                        'contribution_percentage': c.contribution_percentage,
                        'roi': c.roi,
                        'total_spend': c.total_spend,
                        'adstock_decay': c.adstock_decay,
                        'saturation_k': c.saturation_k
                    }
                    for c in results.channel_contributions
                ]
            })

            # Add optimization results if available
            context = None
            if st.session_state.get('optimization_results'):
                opt = st.session_state.optimization_results
                context = f"Optimization results: {opt.improvement_percentage:.1f}% improvement possible"

            request = QARequest(
                question=question,
                model_results_json=model_json,
                context=context
            )

            response = service.answer_question(request)
            return response.answer

        else:
            # General MMM questions without model results
            request = QARequest(
                question=question,
                model_results_json=json.dumps({
                    "note": "No model trained yet. Answering based on general MMM knowledge."
                })
            )

            response = service.answer_question(request)
            return response.answer

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


def generate_summary():
    """Generate a model summary."""
    from ...services.insight_service import InsightService, InsightRequest

    try:
        results = st.session_state.model_results
        model_json = json.dumps({
            'r_squared': results.r_squared,
            'mape': results.mape,
            'base_sales': results.base_sales,
            'base_sales_percentage': results.base_sales_percentage,
            'total_marketing_contribution': results.total_marketing_contribution,
            'channel_contributions': [
                {
                    'channel': c.channel,
                    'contribution_percentage': c.contribution_percentage,
                    'roi': c.roi,
                    'total_spend': c.total_spend
                }
                for c in results.channel_contributions
            ]
        })

        service = InsightService()
        request = InsightRequest(
            model_results_json=model_json,
            insight_type="summary"
        )

        with st.spinner("Generating summary..."):
            response = service.generate_insight(request)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"## Model Summary\n\n{response.content}"
        })
        st.rerun()

    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")


def generate_recommendations():
    """Generate recommendations."""
    from ...services.insight_service import InsightService, InsightRequest

    try:
        results = st.session_state.model_results
        model_json = json.dumps({
            'r_squared': results.r_squared,
            'channel_contributions': [
                {
                    'channel': c.channel,
                    'contribution_percentage': c.contribution_percentage,
                    'roi': c.roi,
                    'total_spend': c.total_spend
                }
                for c in results.channel_contributions
            ]
        })

        service = InsightService()
        request = InsightRequest(
            model_results_json=model_json,
            insight_type="recommendations"
        )

        with st.spinner("Generating recommendations..."):
            response = service.generate_insight(request)

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"## Recommendations\n\n{response.content}"
        })
        st.rerun()

    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")


def generate_report():
    """Generate a full report."""
    from ...services.insight_service import InsightService, ReportRequest

    try:
        results = st.session_state.model_results
        model_json = json.dumps({
            'r_squared': results.r_squared,
            'mape': results.mape,
            'base_sales': results.base_sales,
            'base_sales_percentage': results.base_sales_percentage,
            'total_marketing_contribution': results.total_marketing_contribution,
            'channel_contributions': [
                {
                    'channel': c.channel,
                    'contribution_percentage': c.contribution_percentage,
                    'roi': c.roi,
                    'total_spend': c.total_spend,
                    'adstock_decay': c.adstock_decay
                }
                for c in results.channel_contributions
            ]
        })

        opt_json = None
        if st.session_state.get('optimization_results'):
            opt = st.session_state.optimization_results
            opt_json = json.dumps({
                'improvement_percentage': opt.improvement_percentage,
                'allocations': [
                    {'channel': a.channel, 'change_percentage': a.change_percentage}
                    for a in opt.allocations
                ]
            })

        service = InsightService()
        request = ReportRequest(
            model_results_json=model_json,
            optimization_results_json=opt_json,
            include_sections=["executive_summary", "channel_performance", "recommendations"]
        )

        with st.spinner("Generating report (this may take a minute)..."):
            response = service.generate_report(request)

        report_content = f"# {response.title}\n\n"
        for section_name, section_content in response.sections.items():
            report_content += f"## {section_name.replace('_', ' ').title()}\n\n{section_content}\n\n"

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": report_content
        })
        st.rerun()

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
