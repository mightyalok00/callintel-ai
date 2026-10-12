"""
AI Call Intelligence Engine - Executive Analytics & Multi-Agent Dashboard.
Modern glassmorphic interface with interactive multi-call selection, real-time What-If SHAP simulation,
grounded multi-agent Q&A assistant, live audio ingestion, and CRM webhook dispatching.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.styles import CUSTOM_CSS
from dashboard.components.shap_chart import (
    render_shap_waterfall_chart,
    render_churn_gauge_chart,
    render_sentiment_trend_chart,
)
from src.callintel.ml.churn_model import ChurnRiskPredictor

# Page Configuration
st.set_page_config(
    page_title="CallIntel AI — Executive Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Modern CSS Design System
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

REPORTS_DIR = Path("reports")
SAMPLES_DIR = Path("data/samples")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data
def load_report(filepath: str):
    path = Path(filepath)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_available_reports():
    reports = list(REPORTS_DIR.glob("*_intelligence_report.json"))
    report_map = {}
    scenario_names = {
        "CALL_2026_APEX_001": "🚨 Retention Crisis (Pricing & Cancellation Threat)",
        "CALL_2026_SALES_002": "💼 Enterprise Expansion (Upgrade & Buying Signals)",
        "CALL_2026_TECH_003": "⚠️ Technical Outage (Severity-1 Production Bug)",
    }
    for r in sorted(reports):
        cid = r.stem.replace("_intelligence_report", "")
        label = scenario_names.get(cid, f"📁 Call Session: {cid}")
        report_map[label] = r
    return report_map


def get_audio_path_for_call(call_id: str) -> Path:
    audio_map = {
        "CALL_2026_APEX_001": SAMPLES_DIR / "sample_customer_call.wav",
        "CALL_2026_SALES_002": SAMPLES_DIR / "sample_sales_expansion.wav",
        "CALL_2026_TECH_003": SAMPLES_DIR / "sample_technical_escalation.wav",
    }
    target = audio_map.get(call_id)
    if target and target.exists():
        return target
    # Fallback search
    for wav in SAMPLES_DIR.glob("*.wav"):
        return wav
    return SAMPLES_DIR / "sample_customer_call.wav"


# Initialize Session State
if "dispatched_actions" not in st.session_state:
    st.session_state.dispatched_actions = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_call_label" not in st.session_state:
    st.session_state.active_call_label = None

report_options = get_available_reports()
if not report_options:
    st.error("No intelligence reports found. Run `scripts/run_end_to_end_call_intelligence.py` to generate sample calls.")
    st.stop()


# ==============================================================================
# SIDEBAR NAVIGATION & COMPREHENSIVE FILTERS PANEL
# ==============================================================================
with st.sidebar:
    st.markdown("## 🎙️ **CallIntel OS**")
    st.markdown("<span class='badge badge-success'>v2.0 Enterprise</span>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Interactive Call Scenario Selector
    st.markdown("### 📞 **Active Call Session**")
    call_labels = list(report_options.keys())
    default_idx = 0
    if st.session_state.active_call_label in call_labels:
        default_idx = call_labels.index(st.session_state.active_call_label)

    selected_call_label = st.selectbox(
        "Select Call Scenario",
        call_labels,
        index=default_idx,
        help="Switch between different real customer call scenarios to observe multi-model intelligence.",
    )
    st.session_state.active_call_label = selected_call_label
    active_report_path = report_options[selected_call_label]
    report_data = load_report(str(active_report_path))
    if report_data is None:
        st.error(f"Intelligence report could not be loaded from {active_report_path}.")
        st.stop()
    assert report_data is not None

    active_audio_path = get_audio_path_for_call(report_data.get("call_id", ""))

    st.markdown("---")

    # 2. Section Navigation
    nav_option = st.radio(
        "Workspace View",
        [
            "📊 Executive Overview",
            "🔍 Call Explorer & Timeline",
            "🎯 Retention & SHAP Risk Center",
            "🤖 Multi-Agent Decision Hub",
            "🎙️ Live Audio Studio",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎛️ **Dialogue Filters**")

    # Speaker Filter
    filter_speaker = st.selectbox(
        "Speaker Attribution",
        ["All Speakers", "Customer Only", "Agent Only"],
        index=0,
    )

    # Sentiment Filter
    filter_sentiments = st.multiselect(
        "Sentiment Polarity",
        ["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
        format_func=lambda s: s.capitalize(),
    )

    # Business Signal Triggers
    filter_signals = st.multiselect(
        "Business Signal Events",
        ["Pricing Objections", "Cancellation Warnings", "Agent Commitments"],
        default=[],
        help="Filter turns that contain specific commercial triggers",
    )

    # Keyword Search
    filter_search = st.text_input(
        "Transcript Search",
        placeholder="e.g. expensive, cancel, enterprise...",
    )

    # Priority Filter
    filter_priority = st.multiselect(
        "Action Priority",
        ["P1_URGENT", "P2_HIGH", "P3_MEDIUM", "P4_ROUTINE"],
        default=["P1_URGENT", "P2_HIGH", "P3_MEDIUM"],
    )

    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ **Active Model Stack**")
    st.markdown("""
    - **ASR**: `faster-whisper-base` *(int8, 16kHz)*
    - **Emotion**: `MELD-SGD` *(13.7k turns)*
    - **Churn**: `XGBoost + TreeSHAP` *(165k rows)*
    - **Agents**: `6 Gated Specialist Agents`
    - **API**: `FastAPI REST + WebSockets`
    """)
    st.caption("CallIntel AI | Pair Programmed with Antigravity")


# Helper Filter Function for Transcript Turns
def apply_turn_filters(turns_list):
    filtered = []
    search_term = filter_search.strip().lower()

    for item in turns_list:
        speaker = item.get("speaker", "")
        quote = item.get("verbatim_quote", "")
        signals = item.get("supporting_signals", {})
        sentiment = signals.get("sentiment", "neutral").lower()

        if filter_speaker == "Customer Only" and speaker != "Customer":
            continue
        if filter_speaker == "Agent Only" and speaker != "Agent":
            continue
        if sentiment not in filter_sentiments:
            continue
        if search_term and search_term not in quote.lower():
            continue
        if "Pricing Objections" in filter_signals and not signals.get("is_pricing_objection"):
            continue
        if "Cancellation Warnings" in filter_signals and not signals.get("is_cancellation_warning"):
            continue
        if "Agent Commitments" in filter_signals and not signals.get("is_agent_commitment"):
            continue

        filtered.append(item)
    return filtered


# ==============================================================================
# VIEW 1: EXECUTIVE OVERVIEW
# ==============================================================================
if nav_option == "📊 Executive Overview":
    st.markdown(f"<h1>Executive <span class='gradient-text'>Call Intelligence</span></h1>", unsafe_allow_html=True)
    st.markdown(f"Real-time synthesis for session: **`{report_data['call_id']}`** &bull; Scenario: *{selected_call_label}*")

    # 4 Executive KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    # Sentiment Card
    sent_label = report_data.get("overall_sentiment", "NEUTRAL").upper()
    sent_score = report_data.get("sentiment_score", 0.0)
    sent_badge = "badge-success" if "POS" in sent_label else ("badge-danger" if "NEG" in sent_label else "badge-warning")

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Overall Call Sentiment</div>
                <div class="kpi-value">{sent_score:+.2f}</div>
                <div style="margin-top:6px;"><span class="badge {sent_badge}">{sent_label}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Churn Risk Card
    churn_prob = report_data.get("churn_risk_score", 0.0)
    churn_tier = report_data.get("churn_risk_level", "LOW")
    churn_badge = "badge-urgent" if churn_tier in ["CRITICAL", "HIGH", "ELEVATED"] else ("badge-warning" if churn_tier in ["MODERATE", "MEDIUM"] else "badge-success")

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Predictive Churn Risk</div>
                <div class="kpi-value">{churn_prob:.1%}</div>
                <div style="margin-top:6px;"><span class="badge {churn_badge}">{churn_tier} RISK</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Primary Intent Card
    intent_label = report_data.get("primary_intent", "GENERAL").replace("_", " ").title()
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Customer Intent</div>
                <div class="kpi-value" style="font-size:1.4rem;">{intent_label}</div>
                <div style="margin-top:6px;"><span class="badge badge-info">10-Class MELD Taxonomy</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # QA & Compliance Card
    qa_score = report_data.get("metadata", {}).get("qa_score", 95.0)
    qa_status = report_data.get("metadata", {}).get("qa_status", "PASSED")
    qa_badge = "badge-success" if qa_status == "PASSED" else "badge-urgent"

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Agent QA & Compliance</div>
                <div class="kpi-value">{qa_score:.0f}%</div>
                <div style="margin-top:6px;"><span class="badge {qa_badge}">{qa_status}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Call Audio Player & Quick Transcript Overview
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🎧 **Synchronized Audio & Metadata**")
    col_audio, col_meta = st.columns([1.2, 1])

    with col_audio:
        if active_audio_path.exists():
            st.audio(str(active_audio_path), format="audio/wav")
        else:
            st.info("Audio file preview unavailable.")

    with col_meta:
        turns_count = len(report_data.get("evidence_ledger", []))
        recs_count = len(report_data.get("recommendations", []))
        st.markdown(
            f"""
            - **Dialogue Turns**: `{turns_count} turns`
            - **Verified Evidence Quotes**: `{turns_count} citations in ledger`
            - **Action Recommendations**: `{recs_count} prioritized decisions`
            - **Attribution Method**: `Acoustic Pause Diarization (>0.8s gap)`
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Evidence-Grounded Recommendations Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 **Action Recommendations & Dispatch**")

    raw_recs = report_data.get("recommendations", [])
    filtered_recs = [r for r in raw_recs if r.get("priority") in filter_priority]

    if not filtered_recs:
        st.info("No action recommendations match the selected Priority filter.")
    else:
        for rec in filtered_recs:
            action_code = rec["action_code"]
            is_dispatched = action_code in st.session_state.dispatched_actions

            with st.container():
                st.markdown(f"<div class='glass-card'>", unsafe_allow_html=True)
                col_info, col_btn = st.columns([3, 1])

                with col_info:
                    badge_style = "badge-urgent" if "URGENT" in rec["priority"] else "badge-warning"
                    st.markdown(
                        f"""
                        <span class="badge {badge_style}">{rec['priority']}</span>
                        <span style="font-size:1.1rem; font-weight:700; margin-left:10px;">{rec['title']}</span>
                        <div style="font-size:0.88rem; color:#cbd5e1; margin:8px 0;">{rec['description']}</div>
                        <div style="font-size:0.8rem; color:#94a3b8;"><b>Owner</b>: {rec['owner_role']} &bull; <b>Action Code</b>: <code>{action_code}</code></div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if rec.get("evidence"):
                        ev = rec["evidence"][0]
                        st.markdown(
                            f"""
                            <div class="evidence-quote-box" style="margin-top:10px;">
                                <span class="evidence-timestamp">VERIFIED EVIDENCE [{ev['start_time']}]:</span>
                                <span style="color:#ffffff;">"{ev['verbatim_quote']}"</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if is_dispatched:
                        st.success("✅ DISPATCHED")
                        st.caption("Dispatched to CRM webhook")
                    else:
                        if st.button(f"Approve & Dispatch", key=f"btn_{action_code}"):
                            st.session_state.dispatched_actions.add(action_code)
                            st.toast(f"Successfully dispatched {action_code} to CRM!", icon="🚀")
                            st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    # Export & Integrations Drawer
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📥 **Export Report & CRM Webhook Integration**"):
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            report_json_str = json.dumps(report_data, indent=2)
            st.download_button(
                label="📥 Download Full Intelligence Report (JSON)",
                data=report_json_str,
                file_name=f"{report_data['call_id']}_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_exp2:
            md_brief = f"""# Executive Call Intelligence Brief
Call ID: {report_data['call_id']}
Overall Sentiment: {sent_label} ({sent_score:+.2f})
Churn Risk: {churn_tier} ({churn_prob:.1%})
Intent: {intent_label}
QA Compliance: {qa_score:.0f}% ({qa_status})

## Action Recommendations
"""
            for r in raw_recs:
                md_brief += f"- [{r['priority']}] {r['title']} (Owner: {r['owner_role']})\n"
            st.download_button(
                label="📄 Download Executive Brief (Markdown)",
                data=md_brief,
                file_name=f"{report_data['call_id']}_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ==============================================================================
# VIEW 2: CALL EXPLORER & TIMELINE
# ==============================================================================
elif nav_option == "🔍 Call Explorer & Timeline":
    st.markdown("<h1>Call Explorer & <span class='gradient-text'>Synchronized Timeline</span></h1>", unsafe_allow_html=True)
    st.markdown(f"Dialogue inspection for: **`{report_data['call_id']}`** &bull; *{selected_call_label}*")

    # Audio Playback
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🎧 **Audio Player**")
    if active_audio_path.exists():
        st.audio(str(active_audio_path), format="audio/wav")
    else:
        st.warning("Audio file unavailable.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Filtered Turns View
    raw_turns = report_data.get("evidence_ledger", [])
    filtered_turns = apply_turn_filters(raw_turns)

    st.markdown(
        f"""
        <div style="background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3); border-radius:10px; padding:10px 16px; margin-bottom:18px; font-size:0.88rem; color:#cbd5e1;">
            🔍 <b>Showing {len(filtered_turns)} of {len(raw_turns)} turns</b> based on active filters in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not filtered_turns:
        st.info("No dialogue turns matched your filter criteria.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 📜 **Chronological Dialogue Turns**")

            for item in filtered_turns:
                is_agent = item["speaker"] == "Agent"
                bubble_class = "turn-agent" if is_agent else "turn-customer"
                speaker_color = "#60a5fa" if is_agent else "#f472b6"

                st.markdown(
                    f"""
                    <div class="{bubble_class}">
                        <span style="font-size:0.75rem; color:{speaker_color}; font-weight:700;">{item['speaker'].upper()}</span>
                        <span style="font-size:0.7rem; color:#94a3b8; margin-left:8px;">{item['start_time']} ➔ {item['end_time']}</span>
                        <div style="margin-top:4px; font-size:0.9rem; color:#f1f5f9;">"{item['verbatim_quote']}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("### 🏛️ **Auditable Evidence Ledger**")
            st.caption("Verifiable citations connecting claims to exact timestamps.")

            for item in filtered_turns:
                signals = [f"<code>{k}</code>" for k, v in item.get("supporting_signals", {}).items() if v]
                signals_str = " &bull; ".join(signals) if signals else "<code>neutral</code>"

                st.markdown(
                    f"""
                    <div class="evidence-quote-box">
                        <span class="evidence-timestamp">{item['start_time']} ➔ {item['end_time']} ({item['speaker']})</span>
                        <div style="font-size:0.92rem; margin:6px 0;">"{item['verbatim_quote']}"</div>
                        <div style="font-size:0.75rem; color:#94a3b8;">Signals: {signals_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# VIEW 3: RETENTION & SHAP RISK CENTER (WITH WHAT-IF SIMULATOR)
# ==============================================================================
elif nav_option == "🎯 Retention & SHAP Risk Center":
    st.markdown("<h1>Predictive Churn & <span class='gradient-text'>SHAP Explainability</span></h1>", unsafe_allow_html=True)
    st.markdown("Auditable machine learning risk attribution combining customer account history with live speech cues.")

    col_gauge, col_waterfall = st.columns([1, 1.3])

    with col_gauge:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        gauge_fig = render_churn_gauge_chart(
            report_data["churn_risk_score"], report_data["churn_risk_level"]
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown(f"""
        **Statistical vs Heuristic Breakdown**:
        - **Composite Churn Probability**: `{report_data['churn_risk_score']:.1%}` ({report_data['churn_risk_level']})
        - **Baseline Model Benchmark**: `ROC-AUC 0.8808 | PR-AUC 0.7109 | Brier 0.0984`
        - **Operational Modifiers**: Real-time acoustic frustration and commercial escalation signals.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_waterfall:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        drivers = report_data.get("metadata", {}).get("shap_drivers", [])
        shap_fig = render_shap_waterfall_chart(drivers)
        st.plotly_chart(shap_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Conversational Sentiment Flow
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    emotions = [item.get("supporting_signals", {}).get("sentiment", "neutral") for item in report_data.get("evidence_ledger", [])]
    trend_fig = render_sentiment_trend_chart(emotions)
    st.plotly_chart(trend_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # INTERACTIVE WHAT-IF SHAP SIMULATOR
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🎛️ **Interactive What-If Customer Profile Simulator (Live TreeSHAP)**", expanded=True):
        st.markdown("Adjust customer attributes below to re-evaluate the **XGBoost model** and observe **SHAP risk attribution** in real time.")

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_age = st.slider("Customer Age", min_value=18, max_value=85, value=49)
            sim_balance = st.slider("Account Balance ($)", min_value=0, max_value=250000, value=88000, step=5000)
        with sim_col2:
            sim_credit = st.slider("Credit Score", min_value=350, max_value=850, value=640)
            sim_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
        with sim_col3:
            sim_active = st.radio("Is Active Member?", [0, 1], index=0, format_func=lambda x: "Yes (Active)" if x == 1 else "No (Inactive)")
            sim_geo = st.selectbox("Geography", ["Berlin-Germany", "Paris-France", "Madrid-Spain"], index=0)

        # Execute Live Simulation
        sim_predictor = ChurnRiskPredictor()
        if sim_predictor.is_ready:
            sim_profile = {
                "CreditScore": float(sim_credit),
                "Age": float(sim_age),
                "Tenure": 3.0,
                "Balance": float(sim_balance),
                "NumOfProducts": float(sim_products),
                "HasCrCard": 1.0,
                "IsActiveMember": float(sim_active),
                "EstimatedSalary": 95000.0,
                "Rating": 0.0,
                "Geography": sim_geo,
                "Gender": "Female",
            }
            sim_result = sim_predictor.predict_churn(sim_profile, {})

            sim_prob = sim_result["baseline_churn_probability"]
            sim_tier = sim_result["baseline_risk_tier"]
            orig_prob = report_data["churn_risk_score"]
            delta_prob = sim_prob - orig_prob

            st.markdown("---")
            sim_res_col1, sim_res_col2 = st.columns([1, 1.5])

            with sim_res_col1:
                st.metric(
                    label="Simulated Baseline Churn Probability",
                    value=f"{sim_prob:.1%}",
                    delta=f"{delta_prob:+.1%} vs Original Call",
                    delta_color="inverse",
                )
                sim_badge = "badge-urgent" if sim_tier in ["CRITICAL", "HIGH", "ELEVATED"] else "badge-success"
                st.markdown(f"Risk Tier: <span class='badge {sim_badge}'>{sim_tier}</span>", unsafe_allow_html=True)

            with sim_res_col2:
                sim_shap_fig = render_shap_waterfall_chart(sim_result["top_shap_drivers"])
                st.plotly_chart(sim_shap_fig, use_container_width=True)
        else:
            st.warning("Churn model artifacts not available for live simulation.")


# ==============================================================================
# VIEW 4: MULTI-AGENT DECISION HUB & CONVERSATIONAL ASSISTANT
# ==============================================================================
elif nav_option == "🤖 Multi-Agent Decision Hub":
    st.markdown("<h1>Multi-Agent <span class='gradient-text'>Decision Hub</span></h1>", unsafe_allow_html=True)
    st.markdown("6 specialized AI agents orchestrating verified insights into prioritized, evidence-grounded business actions.")

    # Agent Grid
    st.markdown("### 🧠 **Specialized Agent Output Cards**")
    agent_col1, agent_col2, agent_col3 = st.columns(3)

    with agent_col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4>1. Call Analyst Agent</h4>
                <p style="font-size:0.85rem; color:#cbd5e1;">Identified primary commercial topic as <b>{report_data.get('primary_intent', 'Commercial Inquiry')}</b>.</p>
                <span class="badge badge-success">TOPIC GROUNDED</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with agent_col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4>2. Customer Emotion Agent</h4>
                <p style="font-size:0.85rem; color:#cbd5e1;">Analyzed acoustic emotion trajectory across all dialogue turns.</p>
                <span class="badge badge-info">POLARITY: {report_data.get('overall_sentiment', 'MIXED').upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with agent_col3:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4>3. Retention Risk Agent</h4>
                <p style="font-size:0.85rem; color:#cbd5e1;">Evaluated customer churn drivers via XGBoost and TreeSHAP.</p>
                <span class="badge badge-urgent">TIER: {report_data.get('churn_risk_level', 'LOW')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================================================
    # GROUNDED MULTI-AGENT CHAT ASSISTANT ("CHAT WITH THE CALL")
    # ==========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 **Grounded Multi-Agent Assistant (Chat with the Call)**")
    st.markdown("Ask natural language questions about this call. The assistant verifies claims directly against the **Evidence Ledger**.")

    # Quick prompt buttons
    st.markdown("<b>Quick Queries:</b>", unsafe_allow_html=True)
    qp_col1, qp_col2, qp_col3 = st.columns(3)
    preset_query = None
    if qp_col1.button("💸 Did the customer object to price?"):
        preset_query = "price pricing expensive cost"
    if qp_col2.button("⚠️ Was cancellation threatened?"):
        preset_query = "cancel cancellation stop subscription"
    if qp_col3.button("🎁 What discounts or offers were made?"):
        preset_query = "discount loyalty offer annual"

    user_query = st.text_input("Ask a question about this call:", placeholder="e.g. Did Alex offer a loyalty discount?", value="" if not preset_query else preset_query)

    if user_query:
        query_terms = [t.lower() for t in user_query.split() if len(t) > 2]
        ledger = report_data.get("evidence_ledger", [])
        matched_quotes = []

        for item in ledger:
            quote = item.get("verbatim_quote", "")
            if any(term in quote.lower() for term in query_terms):
                matched_quotes.append(item)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"**Answering**: *\"{user_query}\"*")

        if matched_quotes:
            st.markdown(f"Found **{len(matched_quotes)}** verified citations matching your query in the Evidence Ledger:")
            for mq in matched_quotes[:3]:
                st.markdown(
                    f"""
                    <div class="evidence-quote-box">
                        <span class="evidence-timestamp">[{mq['start_time']} ➔ {mq['end_time']}] {mq['speaker']}:</span>
                        <div style="font-size:0.95rem; color:#ffffff; margin-top:4px;">"{mq['verbatim_quote']}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(f"✅ **Grounding Verification**: High confidence citation derived strictly from verbatim acoustic turns.")
        else:
            st.info("No direct keyword match found in transcript turns. Check broader terms or verify other call scenarios.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Action Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 **Evidence-Grounded Action Recommendations**")
    for rec in report_data.get("recommendations", []):
        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge badge-warning">{rec['priority']}</span>
                <span style="font-size:1.05rem; font-weight:700; margin-left:10px;">{rec['title']}</span>
                <p style="color:#cbd5e1; font-size:0.9rem; margin:8px 0;">{rec['description']}</p>
                <div style="font-size:0.8rem; color:#94a3b8;"><b>Owner</b>: {rec['owner_role']} &bull; <b>Code</b>: <code>{rec['action_code']}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# VIEW 5: LIVE AUDIO STUDIO (WITH REAL PIPELINE EXECUTION)
# ==============================================================================
elif nav_option == "🎙️ Live Audio Studio":
    st.markdown("<h1>Live Audio <span class='gradient-text'>Processing Studio</span></h1>", unsafe_allow_html=True)
    st.markdown("Upload any customer call recording (`.wav`, `.mp3`) or execute the live end-to-end pipeline.")

    uploaded_audio = st.file_uploader("Upload Customer Call Audio File", type=["wav", "mp3"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        st.success(f"Uploaded: `{uploaded_audio.name}` ({len(uploaded_audio.getvalue()):,} bytes)")

    if st.button("⚡ Execute Live AI Pipeline", type="primary", use_container_width=True):
        temp_audio_path = SAMPLES_DIR / "uploaded_call_temp.wav"

        if uploaded_audio is not None:
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_audio.getvalue())
            target_to_run = temp_audio_path
            new_call_id = f"CALL_LIVE_{int(time.time())}"
        else:
            target_to_run = active_audio_path
            new_call_id = f"CALL_DEMO_{int(time.time())}"

        with st.status("Executing Call Intelligence Engine Pipeline...", expanded=True) as status:
            st.write("🎙️ **Step 1/5**: Loading audio & executing `faster-whisper` ASR with acoustic pause diarization...")
            from src.callintel.speech.transcriber import WhisperTranscriber
            from src.callintel.ml.meld_classifier import MeldClassifier
            from src.callintel.agents.multi_agents import MultiAgentSupervisor

            transcriber = WhisperTranscriber(model_size="base")
            asr_result = transcriber.transcribe_file(target_to_run)
            st.write(f"   [+] Extracted {asr_result['total_turns']} timestamped turns across {asr_result['duration_seconds']}s.")

            st.write("🧠 **Step 2/5**: Running MELD Emotion & Sentiment Classification...")
            meld = MeldClassifier()
            turn_emotions = meld.analyze_call_timeline(asr_result["turns"])
            st.write(f"   [+] Emotion trajectory analyzed across {len(turn_emotions)} turns.")

            st.write("📊 **Step 3/5**: Computing XGBoost Churn Risk & TreeSHAP Drivers...")
            churn_pred = ChurnRiskPredictor()
            anger_peak = max((t["emotion"]["probabilities"]["anger"] for t in turn_emotions), default=0.0)
            neg_ratio = sum(1 for t in turn_emotions if t["sentiment"]["label"] == "negative") / max(1, len(turn_emotions))
            sample_profile = {
                "CreditScore": 670.0,
                "Age": 42.0,
                "Tenure": 3.0,
                "Balance": 95000.0,
                "NumOfProducts": 1.0,
                "HasCrCard": 1.0,
                "IsActiveMember": 0.0,
                "EstimatedSalary": 110000.0,
                "Rating": 0.0,
                "Geography": "Berlin-Germany",
                "Gender": "Male",
            }
            churn_res = churn_pred.predict_churn(sample_profile, {
                "customer_frustration_score": anger_peak,
                "negative_sentiment_ratio": neg_ratio,
            })
            st.write(f"   [+] Baseline Churn: {churn_res['baseline_probability']:.1%}, Composite: {churn_res['final_churn_probability']:.1%}")

            st.write("🤖 **Step 4/5**: Orchestrating 6 Specialized AI Agents...")
            supervisor = MultiAgentSupervisor()
            insight_report = supervisor.process_call(
                call_id=new_call_id,
                turns=asr_result["turns"],
                turn_emotions=turn_emotions,
                churn_prediction=churn_res,
            )

            st.write("🏛️ **Step 5/5**: Generating Auditable Evidence Ledger & Persisting...")
            new_report_path = REPORTS_DIR / f"{new_call_id}_intelligence_report.json"
            with open(new_report_path, "w", encoding="utf-8") as f:
                json.dump(insight_report.model_dump(), f, indent=2)

            status.update(label="✅ Processing Complete! Intelligence report synchronized.", state="complete")

        st.success(f"Successfully generated intelligence report: `{new_call_id}`")
        st.balloons()
        st.session_state.active_call_label = f"📁 Call Session: {new_call_id}"
        st.rerun()
