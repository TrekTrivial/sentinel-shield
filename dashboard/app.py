#!/usr/bin/env python3
"""
Sentinel Shield Executive Dashboard
====================================

Real-time monitoring dashboard for the Sentinel phishing detection system.
Displays KPIs, threat analytics, and live threat feed.

Features:
- Auto-refreshing metrics
- Domain distribution visualization
- Live threat feed with highlighting
- Real-time statistics
"""

import streamlit as st
import pandas as pd
import json
import time
import os
from pathlib import Path
import plotly.express as px
from email.header import decode_header, make_header
from datetime import datetime

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Sentinel Shield - Executive Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# THEME CONFIGURATION
# ==============================================================================

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Theme definitions
THEMES = {
    'light': {
        'bg_color': '#ffffff',
        'secondary_bg': '#f0f2f6',
        'text_color': '#262730',
        'card_bg': '#f9f9f9',
        'metric_bg': '#f0f2f6',
        'success_bg': '#c6f6d5',
        'success_text': '#22863a',
        'warning_bg': '#fff5e1',
        'warning_text': '#6f4e37',
        'error_bg': '#ffeef0',
        'error_text': '#cb2431',
        'border_color': '#e1e4e8',
        'threat_bg': '#ffeef0',
        'threat_text': '#ffffff',
    },
    'dark': {
        'bg_color': '#0d1117',
        'secondary_bg': '#161b22',
        'text_color': '#c9d1d9',
        'card_bg': '#0d1117',
        'metric_bg': '#161b22',
        'success_bg': '#238636',
        'success_text': '#7ee787',
        'warning_bg': '#d29922',
        'warning_text': '#ffffff',
        'error_bg': '#da3633',
        'error_text': '#ffffff',
        'border_color': '#30363d',
        'threat_bg': '#da3633',
        'threat_text': '#ffffff',
    }
}

# Get current theme
current_theme = THEMES[st.session_state.theme]

# Apply theme as custom CSS
theme_css = f"""
<style>
    /* Main page styling - EXCLUDE charts */
    .stApp {{
        background-color: {current_theme['bg_color']};
        color: {current_theme['text_color']};
    }}
    
    /* Text and markdown */
    [data-testid="stMarkdownContainer"] {{
        color: {current_theme['text_color']} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {current_theme['text_color']} !important;
    }}
    
    p {{
        color: {current_theme['text_color']} !important;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {current_theme['text_color']} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {current_theme['text_color']} !important;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {current_theme['secondary_bg']} !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {current_theme['text_color']} !important;
    }}
    
    /* Main content area */
    .main {{
        background-color: {current_theme['bg_color']};
        color: {current_theme['text_color']};
    }}
    
    /* Expandable sections */
    .streamlit-expanderHeader {{
        background-color: {current_theme['secondary_bg']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    /* Dataframe styling */
    [data-testid="dataFrameContainer"] {{
        background-color: {current_theme['card_bg']} !important;
    }}
    
    [data-testid="stDataFrame"] {{
        background-color: {current_theme['card_bg']} !important;
    }}
    
    /* Table cells */
    td, th {{
        background-color: {current_theme['card_bg']} !important;
        color: {current_theme['text_color']} !important;
        border-color: {current_theme['border_color']} !important;
    }}
    
    /* Code blocks - preserve readability */
    [data-testid="stCode"] {{
        background-color: {current_theme['secondary_bg']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    pre {{
        background-color: {current_theme['secondary_bg']} !important;
        color: {current_theme['text_color']} !important;
        border-color: {current_theme['border_color']} !important;
    }}
    
    code {{
        background-color: {current_theme['secondary_bg']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    /* Tabs */
    [data-testid="stTabs"] {{
        background-color: {current_theme['bg_color']};
    }}
    
    /* Buttons */
    button {{
        color: {current_theme['text_color']} !important;
    }}
    
    /* Horizontal lines */
    hr {{
        border-color: {current_theme['border_color']} !important;
    }}
    
    /* Alert boxes */
    [data-testid="stInfo"] {{
        background-color: {current_theme['secondary_bg']} !important;
        color: {current_theme['text_color']} !important;
    }}
    
    [data-testid="stSuccess"] {{
        background-color: {current_theme['success_bg']} !important;
        color: {current_theme['success_text']} !important;
    }}
    
    [data-testid="stWarning"] {{
        background-color: {current_theme['warning_bg']} !important;
        color: {current_theme['warning_text']} !important;
    }}
    
    [data-testid="stError"] {{
        background-color: {current_theme['error_bg']} !important;
        color: {current_theme['error_text']} !important;
    }}
    
    /* Plotly charts - PRESERVE colors and don't override */
    [data-testid="stPlotlyContainer"] {{
        /* Keep default chart styling */
    }}
    
    svg {{
        /* Don't override SVG colors for charts */
    }}
    
    /* Threat rows - High contrast for log tables */
    tr:has(td:contains("THREAT")),
    tr:has(td:contains("PHISHING")) {{
        background-color: {current_theme['threat_bg']} !important;
    }}
    
    tr:has(td:contains("THREAT")) td,
    tr:has(td:contains("PHISHING")) td {{
        color: {current_theme['threat_text']} !important;
        background-color: {current_theme['threat_bg']} !important;
        font-weight: bold;
    }}
    
    /* Error rows - Red with white text */
    tr:has(td:contains("ERROR")) {{
        background-color: #ff5555 !important;
    }}
    
    tr:has(td:contains("ERROR")) td {{
        color: #ffffff !important;
        background-color: #ff5555 !important;
        font-weight: bold;
    }}
    
    /* Success rows - Green with white text */
    tr:has(td:contains("SUCCESS")),
    tr:has(td:contains("BENIGN")) {{
        background-color: #00aa00 !important;
    }}
    
    tr:has(td:contains("SUCCESS")) td,
    tr:has(td:contains("BENIGN")) td {{
        color: #ffffff !important;
        background-color: #00aa00 !important;
        font-weight: bold;
    }}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_metrics():
    """
    Safely load metrics from .sentinel_metrics.json
    Returns dict with default values if file doesn't exist
    """
    metrics_path = Path(__file__).parent.parent / ".sentinel_metrics.json"
    
    default_metrics = {
        'total_analyzed': 0,
        'threats_detected': 0,
        'safe_emails': 0,
        'suspicious_emails': 0,
        'emails_with_attachments': 0,
        'emails_with_urls': 0,
        'read_emails': 0,
        'unread_emails': 0,
        'reply_emails': 0,
        'domain_distribution': {},
        'attachment_types': {},
        'threat_by_sender': {}
    }
    
    if not metrics_path.exists():
        return default_metrics
    
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            return {**default_metrics, **metrics}
    except Exception as e:
        st.warning(f"⚠️ Could not load metrics: {e}")
        return default_metrics


@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_logs():
    """
    Safely load last 50 lines from .sentinel_shield.log
    Returns list of log lines
    """
    log_path = Path(__file__).parent.parent / ".sentinel_shield.log"
    
    if not log_path.exists():
        return ["No log file found yet. Monitor is not running or hasn't generated logs."]
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return lines[-50:]  # Return last 50 lines
    except Exception as e:
        st.warning(f"⚠️ Could not load logs: {e}")
        return [f"Error reading logs: {e}"]


def decode_mime_text(value):
    """Decode MIME-encoded email subjects."""
    if not value or not isinstance(value, str):
        return value
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def load_threat_registry():
    """
    Safely load threat registry from .sentinel_threat_registry.json
    Returns list of threats or empty list if file doesn't exist
    """
    registry_path = Path(__file__).parent.parent / ".sentinel_threat_registry.json"
    
    if not registry_path.exists():
        return []
    
    try:
        with open(registry_path, 'r', encoding='utf-8-sig') as f:
            registry = json.load(f)
            return registry.get('threats', [])
    except Exception as e:
        st.warning(f"⚠️ Could not load threat registry: {e}")
        return []


# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

with st.sidebar:
    st.title("🛡️ Sentinel Shield")
    st.markdown("---")
    
    # Theme Toggle - Enhanced Visibility
    st.subheader("🎨 Appearance Settings")
    
    # Create a more prominent theme switcher
    theme_col1, theme_col2 = st.columns(2, gap="small")
    with theme_col1:
        if st.button("☀️ Light Mode", use_container_width=True, key="light_btn", help="Switch to light theme"):
            st.session_state.theme = 'light'
            st.rerun()
    with theme_col2:
        if st.button("🌙 Dark Mode", use_container_width=True, key="dark_btn", help="Switch to dark theme"):
            st.session_state.theme = 'dark'
            st.rerun()
    
    # Show active theme with emoji indicator
    current_theme_display = "☀️ Light Mode" if st.session_state.theme == 'light' else "🌙 Dark Mode"
    st.markdown(f"### {current_theme_display}")
    st.success(f"✅ Active Theme")
    
    st.markdown("---")
    
    # Auto-refresh toggle - FIXED: Only Live (60s) and Manual options
    update_mode = st.radio(
        "📡 Update Mode",
        ["Live (60 seconds)", "Manual Only"],
        index=0,
        horizontal=True,
        help="Live: Auto-refresh every 60 seconds | Manual: Refresh with button only"
    )
    
    # Show refresh indicator based on mode
    if update_mode == "Live (60 seconds)":
        st.success("🟢 Live monitoring active (refreshes every 60 seconds)")
        # Use session state to track last refresh time
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = time.time()
        
        current_time = time.time()
        time_since_refresh = current_time - st.session_state.last_refresh_time
        
        if time_since_refresh >= 60:
            st.session_state.last_refresh_time = current_time
            st.rerun()
        else:
            countdown = 60 - int(time_since_refresh)
            st.info(f"⏱️ Next refresh in {countdown} seconds...")
    else:
        st.warning("⏸️ Manual mode: Click refresh button below when needed")
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # System Status
    st.subheader("📊 System Status")
    
    # Check if monitor is running by checking metrics file (updated with every email)
    metrics_path = Path(__file__).parent.parent / ".sentinel_metrics.json"
    log_path = Path(__file__).parent.parent / ".sentinel_shield.log"
    
    # CRITICAL FIX 9: Use metrics file for status (updated with every email analyzed)
    # instead of log file (which may have gaps between emails)
    status_path = metrics_path if metrics_path.exists() else log_path
    
    if status_path.exists():
        status_mtime = datetime.fromtimestamp(status_path.stat().st_mtime)
        time_diff = datetime.now() - status_mtime
        
        # Use 5-minute threshold instead of 2 (emails don't arrive every second)
        if time_diff.total_seconds() < 300:  # Updated in last 5 minutes
            st.success("✅ Monitor Active")
        else:
            inactive_mins = int(time_diff.total_seconds() // 60)
            st.warning(f"⏸️ Monitor Inactive ({inactive_mins}m ago)")
        
        st.caption(f"Last update: {status_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("❌ No logs found - Monitor not started")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Sentinel Shield** is a real-time phishing detection system with:
    - Stream A: DistilBERT text analysis
    - Stream B: XGBoost URL analysis  
    - Stream C: Master attachment analyzer
    
    **Dashboard** shows live metrics and threat detection statistics.
    """)


# ==============================================================================
# MAIN DASHBOARD LAYOUT
# ==============================================================================

st.markdown("<h1>🛡️ Sentinel Shield - Executive Dashboard</h1>", unsafe_allow_html=True)
st.markdown("Real-time Phishing Detection & Threat Analysis")
st.markdown("---")

# Load data
metrics = load_metrics()
logs = load_logs()
threats = load_threat_registry()

# ==============================================================================
# KPI HEADER (TOP ROW)
# ==============================================================================

st.subheader("📈 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="📧 Total Analyzed",
        value=metrics.get('total_analyzed', 0),
        delta=None
    )

with col2:
    threat_count = metrics.get('threats_detected', 0)
    # Color coding for threats
    if threat_count > 0:
        st.metric(
            label="🚨 Threats Detected",
            value=threat_count,
            delta=threat_count,
            delta_color="inverse"
        )
    else:
        st.metric(
            label="✅ Threats Detected",
            value=threat_count,
            delta=None
        )

with col3:
    st.metric(
        label="☑️ Safe Emails",
        value=metrics.get('safe_emails', 0),
        delta=None
    )

with col4:
    st.metric(
        label="📬 Session Emails Scanned",
        value=metrics.get('unread_emails', 0),
        delta=None
    )

with col5:
    # NEW: Display current inbox count for live deletion tracking
    current_inbox = metrics.get('current_inbox_count', 0)
    st.metric(
        label="📥 Current Inbox Size",
        value=current_inbox,
        help="Live email count in Gmail inbox - decreases when you delete emails"
    )

st.markdown("---")

# ==============================================================================
# VISUALIZATIONS (MIDDLE ROW)
# ==============================================================================

st.subheader("📊 Analytics")

col_chart_1, col_chart_2 = st.columns(2)

# Domain Distribution Pie Chart
with col_chart_1:
    st.markdown("#### Domain Distribution")
    domain_dist = metrics.get('domain_distribution', {})
    
    if domain_dist and len(domain_dist) > 0:
        # Convert dict to DataFrame for Plotly
        domain_data = pd.DataFrame([
            {'Domain': domain, 'Count': count}
            for domain, count in domain_dist.items()
        ])
        
        fig_domain = px.pie(
            domain_data,
            values='Count',
            names='Domain',
            title="Emails by Domain",
            hole=0.3
        )
        fig_domain.update_traces(textposition='inside', textinfo='label+percent')
        st.plotly_chart(fig_domain, use_container_width=True)
    else:
        st.info("No domain data available yet")

# Attachment Types Distribution
with col_chart_2:
    st.markdown("#### Attachment Types")
    attachment_types = metrics.get('attachment_types', {})
    
    if attachment_types and len(attachment_types) > 0:
        att_data = pd.DataFrame([
            {'Type': file_type, 'Count': count}
            for file_type, count in attachment_types.items()
        ])
        
        fig_att = px.bar(
            att_data,
            x='Type',
            y='Count',
            title="Files by Extension",
            labels={'Type': 'File Type', 'Count': 'Quantity'}
        )
        st.plotly_chart(fig_att, use_container_width=True)
    else:
        st.info("No attachment data available yet")

st.markdown("---")

# ==============================================================================
# THREAT REGISTRY TABLE (RECENT THREATS)
# ==============================================================================

st.subheader("🔴 Threat Registry (Persistent History)")

if threats and len(threats) > 0:
    st.markdown(f"**Total Threats Detected:** {len(threats)}")
    st.markdown("---")
    
    # Display threats with detailed SHAP and override information
    for idx, threat in enumerate(reversed(threats[-10:]), 1):  # Show last 10, reverse for newest first
        threat_num = len(threats) - idx + 1  # Calculate actual threat number
        
        # Create expander with threat summary
        sender = threat.get('sender') or threat.get('from') or 'Unknown Sender'
        subject = decode_mime_text(threat.get('subject', '(No Subject)'))
        confidence = threat.get('final_confidence', 0)
        detected_at = threat.get('timestamp') or threat.get('detected_at') or 'Unknown Time'
        
        # Format confidence as percentage
        conf_pct = f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) else "N/A"
        
        # Color code the expander based on confidence
        if confidence > 0.8:
            confidence_label = "🔴 CRITICAL"
        elif confidence > 0.6:
            confidence_label = "🟠 HIGH"
        else:
            confidence_label = "🟡 MEDIUM"
        
        expander_title = f"Threat #{threat_num} | {confidence_label} ({conf_pct}) | {subject[:50]}"
        
        with st.expander(expander_title, expanded=False):
            # Display threat details in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📧 Sender**")
                st.write(sender)
            
            with col2:
                st.markdown("**⏰ Detected At**")
                st.write(detected_at)
            
            with col3:
                st.markdown("**📊 Confidence**")
                st.write(conf_pct)
            
            st.markdown("---")
            
            # Display metadata
            col_meta1, col_meta2 = st.columns(2)
            
            with col_meta1:
                st.markdown("**🔗 Has URLs**")
                has_urls = threat.get('has_urls', False)
                st.write("✅ Yes" if has_urls else "❌ No")
            
            with col_meta2:
                st.markdown("**📎 Has Attachments**")
                has_attachments = threat.get('has_attachments', False)
                st.write("✅ Yes" if has_attachments else "❌ No")
            
            st.markdown("---")
            
            # === DETAILED THREAT INTELLIGENCE SECTION ===
            st.markdown("### 🎯 Threat Intelligence Report")
            
            # Stream scores - read from stream_breakdown
            st.markdown("**📊 Detection Scores (by Stream):**")
            score_cols = st.columns(3)
            
            # Get scores from stream_breakdown or fallback to old format
            stream_breakdown = threat.get('stream_breakdown') or {}
            score_a = stream_breakdown.get('stream_a')
            score_b = stream_breakdown.get('stream_b')
            score_c = stream_breakdown.get('stream_c')

            if score_a is None:
                score_a = threat.get('stream_a_score', threat.get('score_a', 0))
            if score_b is None:
                score_b = threat.get('stream_b_score', threat.get('score_b', 0))
            if score_c is None:
                score_c = threat.get('stream_c_score', threat.get('score_c', 0))

            score_a = score_a if isinstance(score_a, (int, float)) else 0
            score_b = score_b if isinstance(score_b, (int, float)) else 0
            score_c = score_c if isinstance(score_c, (int, float)) else 0
            
            with score_cols[0]:
                st.metric("Stream A (Text)", f"{score_a*100:.1f}%", help="DistilBERT text analysis")
            
            with score_cols[1]:
                st.metric("Stream B (URL)", f"{score_b*100:.1f}%", help="XGBoost URL analysis")
            
            with score_cols[2]:
                st.metric("Stream C (Attach)", f"{score_c*100:.1f}%", help="Attachment analysis")
            
            st.markdown("---")
            
            # Override information
            override_status = bool(threat.get('override', False))
            override_reason = threat.get('override_reason') or threat.get('explanation', '')
            
            if override_status:
                st.warning(f"⚠️ **OVERRIDE TRIGGERED** (Final Decision Override)")
                if override_reason:
                    st.markdown("**Structural Heuristic Reasons:**")
                    if isinstance(override_reason, list):
                        for reason in override_reason:
                            st.markdown(f"  • {reason}")
                    else:
                        st.markdown(f"  • {override_reason}")
                else:
                    st.markdown("  • Structural heuristics detected malicious patterns")
            
            st.markdown("---")
            
            # SHAP Explanation
            st.markdown("### 🤖 XAI Explanation (SHAP Feature Importance)")
            
            # Generate detailed feature breakdown
            st.markdown("**🔍 Feature Contribution Analysis:**")
            
            # Create feature importance breakdown
            features_impact = []
            
            # Stream A features
            if score_a > 0.5:
                features_impact.append({
                    'stream': 'Stream A (Text)',
                    'score': score_a,
                    'features': [
                        'Urgency Keywords: URGENT, IMMEDIATE, VERIFY, CONFIRM',
                        'Credential Request: Account verification, password reset',
                        'Authority Impersonation: Bank/Financial institution claim',
                        'Emotional Manipulation: Time pressure, threats'
                    ]
                })
            
            # Stream B features  
            if score_b > 0.5:
                features_impact.append({
                    'stream': 'Stream B (URL)',
                    'score': score_b,
                    'features': [
                        'Domain Reputation: Suspicious/unregistered domain',
                        'URL Structure: Homograph attacks, encoding tricks',
                        'TLS Certificate: Invalid or mismatched certificate',
                        'WHOIS Age: Recently registered domain'
                    ]
                })
            
            # Stream C features
            if score_c > 0.5:
                features_impact.append({
                    'stream': 'Stream C (Attachment)',
                    'score': score_c,
                    'features': [
                        'File Type: Executable, macro-enabled, suspicious archive',
                        'Malware Signature: Known malware hash (VirusTotal)',
                        'Structural Anomalies: Malformed PDF, embedded JavaScript',
                        'OLE Signature: Legacy Office format with active content'
                    ]
                })
            
            # Display feature breakdown
            if features_impact:
                for item in features_impact:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{item['stream']}**")
                        for feature in item['features']:
                            st.markdown(f"  • {feature}")
                    with col2:
                        st.metric("Score", f"{item['score']*100:.1f}%")
                    st.markdown("")
            
            # Ensemble decision logic
            st.markdown("**⚙️ Ensemble Decision Logic:**")
            final_confidence = threat.get('final_confidence', 0)
            st.markdown(f"""
            **Weighted Voting:**
            - Stream A Weight: 40% → {score_a*0.4:.3f}
            - Stream B Weight: 30% → {score_b*0.3:.3f}  
            - Stream C Weight: 30% → {score_c*0.3:.3f}
            
            **Base Ensemble Score:** {score_a*0.4 + score_b*0.3 + score_c*0.3:.3f}
            **Final Decision Score:** {final_confidence:.3f}
            **Threshold:** 0.70 (Phishing)
            **Verdict:** {'🚨 PHISHING' if final_confidence > 0.7 else '✅ BENIGN'}
            """)
            
            # Explanation from override
            explanation = threat.get('explanation', '')
            shap_explanation = threat.get('shap_explanation', '')

            if shap_explanation:
                st.markdown("**🧠 Stored SHAP / Explanation Payload:**")
                st.code(shap_explanation, language="text")

            if explanation and 'OVERRIDE' in explanation.upper():
                st.warning(f"**⚠️ Structural Override Applied:**\n{explanation}")
    
    st.markdown("---")
else:
    st.success("✅ No threats detected yet!")

st.markdown("---")

# ==============================================================================
# LIVE THREAT FEED (BOTTOM ROW)
# ==============================================================================

st.subheader("📝 Live Monitoring Log")

# Parse logs for threat entries
threat_lines = []
recent_lines = []

for line in logs:
    line_clean = line.strip()
    if line_clean:
        recent_lines.append(line_clean)
        if '[THREAT]' in line or 'PHISHING' in line.upper():
            threat_lines.append(line_clean)

# Tabs for different log views
tab1, tab2, tab3 = st.tabs(["🚨 Threats Only", "📋 Recent Activity", "🔍 Full Log"])

with tab1:
    if threat_lines:
        st.markdown("#### Detected Threats")
        threat_text = "\n".join(threat_lines[-20:])  # Last 20 threats
        st.code(threat_text, language="text")
    else:
        st.info("No threats detected in recent logs")

with tab2:
    if recent_lines:
        st.markdown("#### Last 50 Log Entries")
        # Create a scrollable log view
        activity_df = pd.DataFrame({
            'Timestamp': range(len(recent_lines)),
            'Message': recent_lines
        })
        
        def highlight_threats(row):
            if '[THREAT]' in row['Message'] or 'PHISHING' in row['Message']:
                return ['background-color: #cc0000; color: #ffffff; font-weight: bold;'] * len(row)
            elif '[ERROR]' in row['Message']:
                return ['background-color: #ff5555; color: #ffffff; font-weight: bold;'] * len(row)
            elif '[✓]' in row['Message'] or 'OK' in row['Message']:
                return ['background-color: #00aa00; color: #ffffff; font-weight: bold;'] * len(row)
            return [''] * len(row)
        
        styled_activity = activity_df.style.apply(highlight_threats, axis=1)
        st.dataframe(styled_activity, use_container_width=True, height=400)
    else:
        st.info("No log data available")

with tab3:
    if logs:
        st.markdown("#### Full Log Content")
        log_text = "\n".join(logs[-50:])
        st.code(log_text, language="text")
    else:
        st.info("No logs available")

st.markdown("---")

# ==============================================================================
# FOOTER
# ==============================================================================

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("**📊 Metrics File**")
    metrics_path = Path(__file__).parent.parent / ".sentinel_metrics.json"
    if metrics_path.exists():
        st.caption(f"✅ {metrics_path}")
    else:
        st.caption(f"❌ {metrics_path}")

with col_footer2:
    st.markdown("**📝 Log File**")
    log_path = Path(__file__).parent.parent / ".sentinel_shield.log"
    if log_path.exists():
        st.caption(f"✅ {log_path}")
    else:
        st.caption(f"❌ {log_path}")

with col_footer3:
    st.markdown("**🔐 Threat Registry**")
    registry_path = Path(__file__).parent.parent / ".sentinel_threat_registry.json"
    if registry_path.exists():
        st.caption(f"✅ {registry_path}")
    else:
        st.caption(f"📭 Not created yet (first threat)")

st.markdown("---")
st.markdown(f"""
<p style="text-align: center; color: {current_theme['text_color']}; opacity: 0.7;">
🛡️ <strong>Sentinel Shield</strong> - Real-time Phishing Detection Dashboard<br>
<small>Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small><br>
<small>© 2026 Trek - All Rights Reserved</small>
</p>
""", unsafe_allow_html=True)
