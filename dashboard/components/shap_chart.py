"""
Plotly Chart Component for SHAP Feature Contributions and Churn Risk Gauge.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go


def render_shap_waterfall_chart(drivers: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders an interactive horizontal bar chart of top SHAP drivers.
    Green = reduces churn risk, Red = increases churn risk.
    """
    if not drivers:
        fig = go.Figure()
        fig.update_layout(title="No SHAP drivers available")
        return fig

    features = [d["feature"].replace("Geography_", "Geo: ") for d in reversed(drivers)]
    impacts = [d["impact"] for d in reversed(drivers)]
    colors = ["#ef4444" if imp > 0 else "#10b981" for imp in impacts]
    texts = [f"+{imp:.2f}" if imp > 0 else f"{imp:.2f}" for imp in impacts]

    fig = go.Figure(
        go.Bar(
            x=impacts,
            y=features,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=texts,
            textposition="auto",
            textfont=dict(color="#ffffff", family="Inter", size=12),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Top Risk Drivers (SHAP Value Contribution)</b>",
            font=dict(family="Outfit", size=16, color="#f8fafc"),
        ),
        xaxis=dict(
            title="Log-Odds Impact on Churn (+ Increases / - Decreases)",
            gridcolor="rgba(255, 255, 255, 0.08)",
            zerolinecolor="rgba(255, 255, 255, 0.2)",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            tickfont=dict(color="#f1f5f9", family="Inter", size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=30),
        height=320,
    )
    return fig


def render_churn_gauge_chart(probability: float, tier: str) -> go.Figure:
    """Renders a circular gauge showing calibrated churn risk."""
    colors = {
        "LOW": "#10b981",
        "MODERATE": "#f59e0b",
        "ELEVATED": "#f97316",
        "CRITICAL": "#ef4444",
    }
    bar_color = colors.get(tier, "#ef4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number=dict(suffix="%", font=dict(family="Outfit", size=36, color="#ffffff")),
            title=dict(
                text=f"<b>Churn Risk Tier: {tier}</b>",
                font=dict(family="Outfit", size=16, color=bar_color),
            ),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#64748b", tickfont=dict(color="#94a3b8")),
                bar=dict(color=bar_color, thickness=0.3),
                bgcolor="rgba(30, 41, 59, 0.5)",
                borderwidth=1,
                bordercolor="rgba(255, 255, 255, 0.1)",
                steps=[
                    dict(range=[0, 25], color="rgba(16, 185, 129, 0.15)"),
                    dict(range=[25, 50], color="rgba(245, 158, 11, 0.15)"),
                    dict(range=[50, 75], color="rgba(249, 115, 22, 0.15)"),
                    dict(range=[75, 100], color="rgba(239, 68, 68, 0.2)"),
                ],
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
    )
    return fig


def render_sentiment_trend_chart(trajectory: List[str]) -> go.Figure:
    """Renders call sentiment shift across turns."""
    score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    scores = [score_map.get(s, 0.0) for s in trajectory]
    turns = list(range(1, len(trajectory) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=turns,
            y=scores,
            mode="lines+markers",
            line=dict(color="#6366f1", width=3, shape="spline"),
            marker=dict(
                size=8,
                color=["#10b981" if s > 0 else "#ef4444" if s < 0 else "#64748b" for s in scores],
            ),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.1)",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Call Sentiment Flow (Turn-by-Turn)</b>",
            font=dict(family="Outfit", size=16, color="#f8fafc"),
        ),
        xaxis=dict(
            title="Conversation Turn #",
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            tickvals=[-1, 0, 1],
            ticktext=["Negative", "Neutral", "Positive"],
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=30),
        height=260,
    )
    return fig
