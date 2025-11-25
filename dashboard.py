import streamlit as st
import pandas as pd
import json
import time
import subprocess
import sys
import uuid
import plotly.express as px
from pathlib import Path
from io import BytesIO

# --- CONFIGURATION ---
LOG_FILE = "demo_logs.jsonl"
RULES_FILE = "edge/rules/active.rules"
SCRIPT_PATH = "edge/replay_batch.py"

st.set_page_config(page_title="LLM BASED IDS | SOC", layout="wide", page_icon="🛡️")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #050505; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #111;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #00ccff;
    }
    label[data-testid="stMetricLabel"] { color: #888; }
    div[data-testid="stMetricValue"] { color: #fff; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #080808; border-right: 1px solid #222; }
    
    /* Terminal / Log Boxes */
    .terminal-box {
        font-family: 'Courier New', monospace;
        background-color: #000;
        color: #0f0;
        padding: 10px;
        border: 1px solid #333;
        height: 300px;
        overflow-y: scroll;
        font-size: 11px;
    }
    
    /* Simulation Log Box (Sidebar) */
    .sim-log-box {
        font-family: 'Courier New', monospace;
        background-color: #111;
        color: #aaa;
        padding: 8px;
        border: 1px solid #444;
        border-radius: 4px;
        height: 150px;
        overflow-y: auto;
        font-size: 10px;
        margin-top: 5px;
        display: flex;
        flex-direction: column-reverse; 
    }
    .sim-entry { border-bottom: 1px solid #222; padding: 2px 0; }
    .sim-time { color: #00ccff; font-weight: bold; }
    
    /* Status Badge */
    .status-badge {
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'sim_process' not in st.session_state:
    st.session_state.sim_process = None
if 'sim_history' not in st.session_state:
    st.session_state.sim_history = [] 

# --- DATA FUNCTIONS ---
def get_data():
    if not Path(LOG_FILE).exists(): return pd.DataFrame()
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        if not lines: return pd.DataFrame()
        data = [json.loads(line) for line in lines]
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def get_rules():
    if not Path(RULES_FILE).exists(): return "System Clean."
    try:
        with open(RULES_FILE, "r") as f: lines = f.readlines()
        valid = [l.strip() for l in lines if "NOOP" not in l and l.strip()]
        return "<br>".join(valid) if valid else "No Active Blocks."
    except: return "Error."

def generate_report(df):
    """Generates a detailed text report for download"""
    if df.empty: return "No data available."
    
    total_attacks = len(df[df['is_attack'] == True])
    if total_attacks > 0:
        top_ip = df[df['is_attack'] == True]['src_ip'].value_counts().idxmax()
    else:
        top_ip = "None"
        
    breakdown = df['label'].value_counts().to_string()
    
    # Header Section
    report = f"""
================================================================================
       LLM BASED IDS - COMPREHENSIVE INCIDENT REPORT
================================================================================
Report Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
Total Packets Scanned: {len(df)}
Total Threats Detected: {total_attacks}
Top Attacker IP: {top_ip}

--- THREAT BREAKDOWN ---
{breakdown}

--- EDGE DEFENSE STATUS ---
Active Rules Deployed: {len(get_rules().split('<br>'))}

================================================================================
                            FULL EVENT LOG STREAM
================================================================================
TIMESTAMP        | SEVERITY | SRC IP           | THREAT LABEL        | ACTION TAKEN
--------------------------------------------------------------------------------
"""
    # Detailed Log Loop
    # We iterate through the dataframe to create the log rows
    for index, row in df.iterrows():
        severity = "CRITICAL" if row['is_attack'] else "INFO"
        # Formatting specifically aligned for text file readability
        line = f"{row['time_str']:<16} | {severity:<8} | {row['src_ip']:<16} | {row['label']:<19} | {row['action']}\n"
        report += line

    report += "\n================================================================================\n"
    report += "*** END OF REPORT ***"
    
    return report

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ SYSTEM CONTROLS")

# Flush Button
if st.sidebar.button("🗑️ FLUSH LOGS & RESET", type="primary"):
    with open(LOG_FILE, 'w') as f: f.write("")
    with open(RULES_FILE, 'w') as f: f.write("")
    st.session_state.sim_history = [] 
    st.toast("System Memory Wiped", icon="🧹")
    time.sleep(0.5)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Parameters")

total_packets = st.sidebar.slider("Total Packets", 100, 10000, 1000, 100)
burst_factor = st.sidebar.slider("Burst Factor", 1, 20, 5)
normal_ratio = st.sidebar.slider("Normal Traffic Ratio", 0.0, 1.0, 0.8)
sleep_time = st.sidebar.slider("Speed (Delay)", 0.0, 1.0, 0.05)

c1, c2 = st.sidebar.columns(2)

# START BUTTON
if c1.button("▶ START"):
    is_running = False
    if st.session_state.sim_process is not None:
        if st.session_state.sim_process.poll() is None:
            is_running = True
            
    if is_running:
        st.toast("Simulation already active!", icon="⚠️")
    else:
        # 1. Start Process
        cmd = [
            sys.executable, SCRIPT_PATH, 
            "--total", str(total_packets), 
            "--burst", str(burst_factor), 
            "--normal-ratio", str(normal_ratio), 
            "--sleep", str(sleep_time)
        ]
        st.session_state.sim_process = subprocess.Popen(cmd)
        
        # 2. Add to Sidebar Log (APPEND TO END)
        t_str = time.strftime("%H:%M:%S")
        log_entry = f"[{t_str}] Pkts={total_packets} | Burst={burst_factor} | Ratio={normal_ratio}"
        st.session_state.sim_history.append(log_entry)
        
        st.toast("Attack Simulation Initiated", icon="🚀")
        st.rerun()

# STOP BUTTON
if c2.button("⏹ STOP"):
    if st.session_state.sim_process:
        st.session_state.sim_process.terminate()
        st.session_state.sim_process = None
        st.toast("Simulation Halted", icon="🛑")
        st.rerun()

# --- NEW FEATURE B: DOWNLOAD EXPANDED REPORT ---
st.sidebar.markdown("---")
df_report = get_data()
if not df_report.empty:
    report_text = generate_report(df_report)
    st.sidebar.download_button(
        label="📄 DOWNLOAD FULL REPORT",
        data=report_text,
        file_name="llm-based_full_report.txt",
        mime="text/plain"
    )

# --- SIDEBAR: SIMULATION HISTORY LOG ---
st.sidebar.markdown("### 📜 Execution Log")

history_html = ""
for item in st.session_state.sim_history:
    parts = item.split("] ", 1)
    if len(parts) == 2:
        ts = parts[0] + "]"
        msg = parts[1]
        history_html += f"<div class='sim-entry'><span class='sim-time'>{ts}</span> {msg}</div>"
    else:
        history_html += f"<div class='sim-entry'>{item}</div>"

st.sidebar.markdown(f"<div class='sim-log-box'>{history_html}</div>", unsafe_allow_html=True)


# --- MAIN DASHBOARD CONTENT ---
st.title("🛡️ LLM BASED IDS: Live Threat Monitor")

# Determine Status for display
status_html = ""
if st.session_state.sim_process is not None:
    if st.session_state.sim_process.poll() is None:
        status_html = "<div class='status-badge' style='background:#222; color:#00ccff; border:1px solid #00ccff;'>⚠️ SIMULATION RUNNING</div>"
    else:
        status_html = "<div class='status-badge' style='background:#111; color:#00ff00; border:1px solid #00ff00;'>✅ DONE</div>"
        st.session_state.sim_process = None 
else:
    status_html = "<div class='status-badge' style='background:#111; color:#666; border:1px dashed #444;'>⚪ READY</div>"

st.markdown(status_html, unsafe_allow_html=True)

# 1. METRICS
df = get_data()
m1, m2, m3, m4 = st.columns(4)

if not df.empty:
    total = len(df)
    threats = len(df[df['is_attack'] == True])
    window = df.tail(50)
    conf = window['confidence'].mean()
    lat = window['latency_ms'].iloc[-1]
    
    m1.metric("Total Packets", total)
    m2.metric("Threats Detected", threats)
    m3.metric("Avg Confidence", f"{conf:.1%}")
    m4.metric("Live Latency", f"{lat:.1f} ms")
else:
    m1.metric("Total Packets", 0)
    m2.metric("Threats Detected", 0)
    m3.metric("Avg Confidence", "0%")
    m4.metric("Live Latency", "0 ms")

# 2. CHARTS
c_left, c_right = st.columns([2, 1])

if not df.empty:
    chart_df = df.tail(60).copy()
    
    color_map = {
        "Normal": "#00FF00", "DDoS_TCP": "#FF0000", "DDoS_UDP": "#FF3333",
        "SQL_injection": "#FFA500", "Ransomware": "#FF00FF", "XSS": "#FFFF00",
        "Port_Scanning": "#00CCFF", "Password": "#AA00AA"
    }
    
    with c_left:
        st.caption("📡 LIVE TRAFFIC SWIMLANE")
        fig_lane = px.scatter(
            chart_df, x="timestamp", y="label", color="label",
            size="latency_ms", size_max=15, color_discrete_map=color_map,
            hover_data=["src_ip", "action"]
        )
        fig_lane.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#111',
            font=dict(color="white"), xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="#333", title=None),
            showlegend=False, uirevision='constant'
        )
        st.plotly_chart(fig_lane, use_container_width=True, key=f"lane_{uuid.uuid4()}")

    with c_right:
        st.caption("🎯 THREAT DISTRIBUTION (Window)")
        attacks_only = chart_df[chart_df['label'] != 'Normal']
        if not attacks_only.empty:
            fig_pie = px.pie(attacks_only, names='label', hole=0.6, color_discrete_sequence=px.colors.qualitative.Bold)
            fig_pie.update_layout(
                height=350, margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"),
                showlegend=True, legend=dict(orientation="h", y=-0.1), uirevision='constant'
            )
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{uuid.uuid4()}")
        else:
            st.info("No Active Threats")
else:
    with c_left: st.info("Waiting for Data...")
    with c_right: st.info("Waiting for Data...")

# 3. LOGS & RULES
c_log, c_rule = st.columns([2, 1])
with c_log:
    st.caption("📋 EVENT LOGS")
    if not df.empty:
        display_cols = ["time_str", "label", "action", "src_ip", "latency_ms"]
        view_df = df[display_cols].iloc[::-1].reset_index(drop=True)
        styled_df = view_df.style.map(
            lambda x: "color: #ff3333" if x != "Normal" else "color: #00ff00",
            subset=["label"]
        )
        st.dataframe(styled_df, use_container_width=True, height=300)

with c_rule:
    st.caption("🔒 ACTIVE EDGE RULES")
    st.markdown(f'<div class="terminal-box">{get_rules()}</div>', unsafe_allow_html=True)


# --- AUTO-REFRESH LOGIC ---
if st.session_state.sim_process is not None:
    if st.session_state.sim_process.poll() is None:
        time.sleep(1)
        st.rerun()
    else:
        st.rerun()