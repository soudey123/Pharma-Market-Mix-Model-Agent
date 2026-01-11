"""
Insight Service for Pharma MMM Agent.

Uses Claude API to generate natural language insights and Q&A.
"""

import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from loguru import logger
import os


# ============================================================================
# Input/Output Contracts
# ============================================================================

class InsightRequest(BaseModel):
    """Request for generating insights."""
    model_results_json: str
    insight_type: str = "summary"  # summary, recommendations, deep_dive
    focus_areas: Optional[List[str]] = None


class InsightResponse(BaseModel):
    """Response with generated insights."""
    insight_type: str
    content: str
    key_findings: List[str]
    recommendations: List[str]


class QARequest(BaseModel):
    """Request for Q&A about model results."""
    question: str
    model_results_json: str
    context: Optional[str] = None


class QAResponse(BaseModel):
    """Response to a Q&A query."""
    question: str
    answer: str
    confidence: str  # high, medium, low
    sources: List[str]


class ReportRequest(BaseModel):
    """Request for generating a full report."""
    model_results_json: str
    optimization_results_json: Optional[str] = None
    include_sections: List[str] = Field(
        default=["executive_summary", "channel_performance", "recommendations"]
    )


class ReportResponse(BaseModel):
    """Generated report."""
    title: str
    sections: Dict[str, str]
    generated_at: str


# ============================================================================
# Insight Service Implementation
# ============================================================================

class InsightService:
    """
    Claude-powered insight generation service.

    Provides natural language summaries, Q&A, and recommendations
    based on MMM results.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Anthropic API key."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                raise
        return self._client

    def _call_claude(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 2000
    ) -> str:
        """Make a call to Claude API."""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"Error generating insight: {str(e)}"

    def generate_insight(self, request: InsightRequest) -> InsightResponse:
        """
        Generate insights from model results.

        Args:
            request: InsightRequest with model results and type

        Returns:
            InsightResponse with generated insights
        """
        logger.info(f"Generating {request.insight_type} insights...")

        results = json.loads(request.model_results_json)

        system_prompt = """You are an expert pharmaceutical marketing analyst specializing in
        Market Mix Modeling (MMM). You provide clear, actionable insights for marketing teams.
        Focus on pharma-specific considerations like:
        - Longer carryover effects compared to CPG
        - Regulatory constraints on DTC advertising
        - HCP engagement effectiveness
        - Conference and medical journal impact

        Provide insights in a professional, data-driven manner."""

        if request.insight_type == "summary":
            user_message = f"""Analyze these Market Mix Model results and provide a summary:

{json.dumps(results, indent=2)}

Please provide:
1. A 2-3 paragraph executive summary
2. Top 3-5 key findings (bullet points)
3. Top 3-5 actionable recommendations

Focus on channel effectiveness, ROI, and budget optimization opportunities."""

        elif request.insight_type == "recommendations":
            user_message = f"""Based on these Market Mix Model results, provide detailed recommendations:

{json.dumps(results, indent=2)}

Please provide:
1. Specific budget reallocation recommendations
2. Channel-specific optimization suggestions
3. Timing and seasonality considerations
4. Risk factors to consider

Be specific with percentages and dollar amounts where possible."""

        elif request.insight_type == "deep_dive":
            focus = request.focus_areas or ["all channels"]
            user_message = f"""Provide a deep-dive analysis of these MMM results, focusing on: {', '.join(focus)}

{json.dumps(results, indent=2)}

Include:
1. Detailed performance analysis
2. Statistical confidence in findings
3. Comparison to industry benchmarks
4. Specific improvement opportunities"""

        else:
            user_message = f"Analyze these MMM results:\n{json.dumps(results, indent=2)}"

        response_text = self._call_claude(system_prompt, user_message)

        # Parse the response to extract structured components
        key_findings = self._extract_bullet_points(response_text, "findings")
        recommendations = self._extract_bullet_points(response_text, "recommendations")

        return InsightResponse(
            insight_type=request.insight_type,
            content=response_text,
            key_findings=key_findings,
            recommendations=recommendations
        )

    def answer_question(self, request: QARequest) -> QAResponse:
        """
        Answer a question about model results.

        Args:
            request: QARequest with question and context

        Returns:
            QAResponse with answer
        """
        logger.info(f"Answering question: {request.question[:50]}...")

        results = json.loads(request.model_results_json)

        system_prompt = """You are an expert pharmaceutical marketing analyst.
        Answer questions about Market Mix Model results clearly and accurately.
        Base your answers only on the provided data.
        If you're not sure about something, say so.
        Provide specific numbers and percentages when available."""

        context_str = f"\nAdditional context: {request.context}" if request.context else ""

        user_message = f"""Based on these Market Mix Model results, please answer the following question.

MMM Results:
{json.dumps(results, indent=2)}
{context_str}

Question: {request.question}

Please provide:
1. A clear, direct answer
2. Supporting data from the results
3. Any caveats or limitations"""

        response_text = self._call_claude(system_prompt, user_message, max_tokens=1000)

        # Determine confidence based on data availability
        confidence = "high"
        if "uncertain" in response_text.lower() or "not sure" in response_text.lower():
            confidence = "low"
        elif "may" in response_text.lower() or "might" in response_text.lower():
            confidence = "medium"

        return QAResponse(
            question=request.question,
            answer=response_text,
            confidence=confidence,
            sources=["MMM Model Results"]
        )

    def generate_report(self, request: ReportRequest) -> ReportResponse:
        """
        Generate a comprehensive report.

        Args:
            request: ReportRequest with results and sections

        Returns:
            ReportResponse with full report
        """
        from datetime import datetime

        logger.info("Generating comprehensive report...")

        results = json.loads(request.model_results_json)
        optimization = json.loads(request.optimization_results_json) if request.optimization_results_json else None

        sections = {}

        system_prompt = """You are an expert pharmaceutical marketing analyst writing a
        professional report on Market Mix Model results. Write in a clear, business-appropriate style.
        Use specific numbers and percentages. Format for easy reading."""

        for section in request.include_sections:
            if section == "executive_summary":
                user_message = f"""Write an executive summary for this MMM analysis:

Results: {json.dumps(results, indent=2)}
{f'Optimization: {json.dumps(optimization, indent=2)}' if optimization else ''}

Keep it to 3-4 paragraphs covering:
- Overall model performance
- Key channel insights
- Main recommendations"""

            elif section == "channel_performance":
                user_message = f"""Write a detailed channel performance section:

Results: {json.dumps(results, indent=2)}

For each channel, discuss:
- Contribution to sales
- ROI and efficiency
- Adstock/carryover characteristics
- Recommendations"""

            elif section == "recommendations":
                user_message = f"""Write a recommendations section:

Results: {json.dumps(results, indent=2)}
{f'Optimization: {json.dumps(optimization, indent=2)}' if optimization else ''}

Include:
- Budget reallocation suggestions
- Channel-specific tactics
- Implementation priorities
- Expected impact"""

            elif section == "methodology":
                user_message = """Write a methodology section explaining:
- Bayesian Market Mix Modeling approach
- Adstock transformations (geometric/Weibull)
- Saturation curves (Hill function)
- Model validation metrics

Keep it accessible for non-technical stakeholders."""

            else:
                continue

            section_content = self._call_claude(system_prompt, user_message, max_tokens=1500)
            sections[section] = section_content

        return ReportResponse(
            title="Pharmaceutical Market Mix Model Analysis Report",
            sections=sections,
            generated_at=datetime.now().isoformat()
        )

    def _extract_bullet_points(self, text: str, section_type: str) -> List[str]:
        """Extract bullet points from text."""
        lines = text.split('\n')
        bullet_points = []

        for line in lines:
            line = line.strip()
            if line.startswith(('-', '•', '*', '1', '2', '3', '4', '5')):
                # Clean up the bullet
                cleaned = line.lstrip('-•*0123456789. ')
                if cleaned:
                    bullet_points.append(cleaned)

        # Return up to 5 items
        return bullet_points[:5]

    @staticmethod
    def generate_channel_summary(
        channel_contributions: List[Dict],
        optimization_results: Optional[Dict] = None
    ) -> str:
        """
        Generate a quick channel summary without API call.

        Useful for n8n workflows where API calls should be minimized.
        """
        summary_lines = ["## Channel Performance Summary\n"]

        # Sort by contribution
        sorted_channels = sorted(
            channel_contributions,
            key=lambda x: x.get('contribution_percentage', 0),
            reverse=True
        )

        for channel in sorted_channels:
            name = channel.get('channel', 'Unknown')
            contrib_pct = channel.get('contribution_percentage', 0)
            roi = channel.get('roi', 0)
            spend = channel.get('total_spend', 0)

            summary_lines.append(f"### {name}")
            summary_lines.append(f"- Contribution: {contrib_pct:.1f}%")
            summary_lines.append(f"- ROI: {roi:.2f}")
            summary_lines.append(f"- Total Spend: ${spend:,.0f}")

            if optimization_results:
                for alloc in optimization_results.get('allocations', []):
                    if alloc.get('channel') == name:
                        change = alloc.get('change_percentage', 0)
                        direction = "increase" if change > 0 else "decrease"
                        summary_lines.append(f"- Recommendation: {direction} by {abs(change):.1f}%")

            summary_lines.append("")

        return "\n".join(summary_lines)
