# Sentinel Shield Dashboard - Quick Start Guide

## 📊 Dashboard Overview

The Sentinel Shield Executive Dashboard provides real-time monitoring and analytics for your phishing detection system.

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  MAIN DASHBOARD              │
├─────────────────────────────┼──────────────────────────────┤
│                             │                              │
│  🔄 Auto-Refresh Toggle     │  🛡️ SENTINEL SHIELD         │
│  ⏱️  Refresh Interval        │  Executive Dashboard        │
│  📊 System Status           │                              │
│  ℹ️ About                   │  ┌──────┬──────┬──────┬──┐  │
│                             │  │ KPI  │ KPI  │ KPI  │KPI│
│  File Status                │  └──────┴──────┴──────┴──┘  │
│  • Metrics ✅               │                              │
│  • Logs ✅                  │  ┌──────────────┬──────────┐ │
│  • Registry 📭              │  │ Domain Pie   │ Attach   │ │
│                             │  │ Chart        │ Bar      │ │
│                             │  └──────────────┴──────────┘ │
│                             │                              │
│                             │  Threat Registry (10 latest) │
│                             │                              │
│                             │  Live Feed Tabs:             │
│                             │  • Threats Only              │
│                             │  • Recent Activity           │
│                             │  • Full Log                  │
│                             │                              │
└─────────────────────────────┴──────────────────────────────┘
```

## 🚀 Getting Started (30 seconds)

### Step 1: Open Terminal (PowerShell)
```powershell
cd "d:\Documents\sem-8\Major Project phase 2\Implement"
```

### Step 2: Install Dashboard Dependencies (one-time)
```powershell
pip install -r dashboard/requirements.txt
```

### Step 3: Start Monitor (Terminal 1)
```powershell
python scripts/inference/9_gmail_live_monitor.py
```

### Step 4: Start Dashboard (Terminal 2)
```powershell
streamlit run dashboard/app.py
```

**Or use the launcher:**
```powershell
.\dashboard\run_dashboard.ps1
```

Dashboard opens at: **http://localhost:8501**

## 📱 Dashboard Sections

### 1️⃣ Sidebar (Left)

**Live Auto-Refresh Toggle**
- Toggle "Live Auto-Refresh" to enable real-time updates
- Updates every 5 seconds automatically
- Great for live monitoring scenarios

**System Status**
- Shows if monitor is active
- Displays last update timestamp
- Color coded: ✅ Active, ⏸️ Inactive, ❌ Not running

**File Status**
- ✅ Metrics file exists
- ✅ Logs file exists
- 📭 Threat registry (created on first threat)

### 2️⃣ KPI Header (Top)

Four metric cards displaying:

```
┌─────────┬──────────┬─────────┬──────────┐
│  📧     │  🚨      │  ☑️     │  📬      │
│ TOTAL   │ THREATS  │ SAFE    │ UNREAD   │
│ 30      │ 3        │ 25      │ 2        │
└─────────┴──────────┴─────────┴──────────┘
```

**Color Coding:**
- 🚨 Threats: RED if > 0 (inverse delta coloring)
- ☑️ Safe: GREEN (normal display)
- 📧 Total: BLUE (informational)
- 📬 Unread: PURPLE (informational)

### 3️⃣ Analytics Charts (Middle)

**Left: Domain Distribution**
- Pie chart showing email breakdown by domain
- Example: gmail.com (15), yahoo.com (8), outlook.com (7)
- Interactive: hover to see percentages

**Right: Attachment Types**
- Bar chart showing file types encountered
- Example: PDF (12), DOC (8), XLS (5), PNG (3)
- Helps identify attachment threat patterns

### 4️⃣ Threat Registry Table

**Persistent threat history** (never resets)

Columns displayed:
- **Sender**: Email address that sent the phishing email
- **Subject**: Email subject line
- **Confidence**: Phishing confidence (0.0-1.0)
- **Detected At**: Timestamp of detection
- **Has URLs**: Whether email contained suspicious URLs
- **Has Attachments**: Whether email contained suspicious files

**Color Coding:**
- Confidence > 0.8: 🔴 RED (high threat)
- Confidence > 0.6: 🟠 ORANGE (medium threat)
- Confidence ≤ 0.6: ⚪ NORMAL (low threat)

**Shows:** Last 10 threats  
**Total Count:** Displayed at bottom

### 5️⃣ Live Feed (Bottom)

Three tabs for different log views:

#### Tab 1: 🚨 Threats Only
- Shows only lines with `[THREAT]` or `PHISHING`
- Last 20 threat entries
- Quick view of detected phishing emails
- Example:
  ```
  [THREAT] attacker@evil.com: Urgent Payment Required
  [THREAT] phisher@fake-bank.com: Verify Your Account
  [THREAT] malware@bad-site.com: Exclusive Offer for You
  ```

#### Tab 2: 📋 Recent Activity
- Last 50 log entries with color highlighting
- Scroll-able table format
- **Red background**: [THREAT] or PHISHING
- **Green background**: [✓] or OK
- **Orange background**: [ERROR]

#### Tab 3: 🔍 Full Log
- Raw log content
- Last 50 lines from `.sentinel_shield.log`
- Searchable text block
- Useful for detailed debugging

## 📊 Data Files

The dashboard reads from your project root:

```
📁 Implement (project root)
├── 📄 .sentinel_metrics.json
│   └─ Metrics: total_analyzed, threats_detected, safe_emails, etc.
│
├── 📄 .sentinel_shield.log
│   └─ Live logs: [THREAT], [ANALYZE], [STARTUP], etc.
│
├── 📄 .sentinel_threat_registry.json
│   └─ Persistent threat history (created on first threat)
│
└── 📁 dashboard/
    ├── app.py                    ← Main dashboard script
    ├── config.json               ← Configuration
    ├── requirements.txt          ← Python dependencies
    ├── run_dashboard.ps1         ← PowerShell launcher
    ├── run_dashboard.bat         ← Batch launcher
    └── README.md                 ← Full documentation
```

## 🔄 How Data Flows

```
Monitor (9_gmail_live_monitor.py)
    ↓
    ├─→ .sentinel_metrics.json (updated each analysis)
    ├─→ .sentinel_shield.log (appended each operation)
    └─→ .sentinel_threat_registry.json (threat recorded)
    ↓
Dashboard (dashboard/app.py)
    ↓
    ├─→ Reads metrics every 5 seconds
    ├─→ Reads logs every 5 seconds
    └─→ Reads threat registry every 5 seconds
    ↓
User sees real-time updates in browser
```

## ⚙️ Configuration

Edit `dashboard/config.json` to customize:

```json
{
  "dashboard": {
    "refresh_interval_seconds": 5  // Change how often data refreshes
  },
  "logs": {
    "max_lines_to_display": 50     // Change how many log lines shown
  },
  "threat_registry": {
    "max_threats_to_display": 10   // Change threat history count
  }
}
```

## 🔗 Keyboard Shortcuts (Streamlit)

| Shortcut | Action |
|----------|--------|
| `R` | Refresh/rerun page |
| `C` | Clear cache |
| `K` | Open keyboard shortcuts |
| `Ctrl+C` (terminal) | Stop dashboard |

## 📊 Interpreting the Dashboard

### What do the metrics mean?

- **Total Analyzed**: All emails scanned by Sentinel
- **Threats Detected**: Emails flagged as PHISHING
- **Safe Emails**: Emails flagged as SAFE
- **Unread Emails**: Emails not yet read by user

### What's a "good" state?

✅ **Healthy:**
- Total Analyzed > 0 (system is working)
- Threats Detected: 0 or low (system is secure)
- Safe Emails: majority of total (most email is legitimate)
- Unread Emails: matches expectations

⚠️ **Watch Out For:**
- Threats Detected increasing rapidly (potential attack)
- All emails marked PHISHING (detector misconfigured)
- No emails analyzed (monitor not running)
- System Inactive > 2 minutes (monitor crashed)

### Domain Distribution

Shows which email providers send you the most mail:
- **gmail.com, outlook.com, yahoo.com**: Common legitimate domains
- **Unknown/rare domains**: Potential spoofing concern
- **Spike in unusual domains**: Possible targeted attack

## 🐛 Troubleshooting

### Dashboard shows "No logs found"
1. Ensure monitor is running: `python scripts/inference/9_gmail_live_monitor.py`
2. Wait 10 seconds for files to be created
3. Refresh browser (R key)

### Dashboard shows "No data available"
1. Check if metrics file exists: `.sentinel_metrics.json`
2. Monitor must run to create this file
3. Dashboard provides sensible defaults (0 values)

### Port 8501 already in use
```powershell
streamlit run dashboard/app.py --server.port 8502
```

### Performance is slow
1. Close auto-refresh toggle (reduces refreshes)
2. Disable live monitoring temporarily
3. Check if monitor is consuming resources

### Threat registry empty
1. This is normal on first run
2. Registry created when first threat detected
3. Zero threats = system is secure!

## 📈 Common Tasks

### Check Total Threats Ever Detected
Look at "Threat Registry" table - shows last 10 threats with full details

### Find Email from Specific Sender
Use browser's Find feature (Ctrl+F) in the Threats tab

### Export Threat History
Open `.sentinel_threat_registry.json` in JSON viewer or text editor

### Clear Dashboard Cache
Press `C` in Streamlit app (clears all caches)

### Check Monitor Status
Look at sidebar "System Status" section

## 🎨 Customization Examples

### Change refresh rate to 10 seconds
Edit `dashboard/app.py` line 72:
```python
@st.cache_data(ttl=10)  # Was ttl=5
```

### Show last 100 log lines instead of 50
Edit `dashboard/app.py` line 110:
```python
return lines[-100:]  # Was lines[-50:]
```

### Hide threat registry section
In `dashboard/app.py`, comment out lines ~250-270

### Add new KPI metric
1. Add column: `col5 = st.columns(5)` (change 4 to 5)
2. Add metric: `st.metric("label", value)`

## 📞 Need Help?

1. Check dashboard README: `dashboard/README.md`
2. Check monitor logs: `.sentinel_shield.log`
3. Check project root README: Check for main documentation

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r dashboard/requirements.txt`
2. ✅ Start monitor: `python scripts/inference/9_gmail_live_monitor.py`
3. ✅ Start dashboard: `streamlit run dashboard/app.py`
4. ✅ Access at: `http://localhost:8501`
5. ✅ Toggle auto-refresh in sidebar
6. ✅ Monitor threats in real-time!

---

**Happy monitoring! 🛡️**
