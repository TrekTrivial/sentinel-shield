# Sentinel Shield Executive Dashboard

Real-time monitoring dashboard for the Sentinel phishing detection system.

## Features

✨ **Live KPIs**
- Total Emails Analyzed
- Threats Detected (highlighted if > 0)
- Safe Emails Count
- Unread Emails Count

📊 **Visualizations**
- Domain Distribution (Pie Chart)
- Attachment Types (Bar Chart)

🔴 **Threat Registry**
- Persistent threat history (never resets)
- Confidence scores from all 3 detection streams
- Sender, subject, timestamp, and explanations

📝 **Live Feed**
- Real-time log monitoring
- Threat-only filter
- Activity timeline
- Full log content

🔄 **Auto-Refresh**
- Toggle live auto-refresh (updates every 5 seconds)
- Configurable refresh intervals
- System status indicator

## Installation

### 1. Install Dependencies

```bash
# From project root
pip install -r dashboard/requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas plotly
```

### 2. Ensure Monitor is Running

The dashboard reads from:
- `.sentinel_metrics.json` (metrics file)
- `.sentinel_shield.log` (logs)
- `.sentinel_threat_registry.json` (threat history)

These files are created by the monitoring script:
```bash
python scripts/inference/9_gmail_live_monitor.py
```

## Running the Dashboard

### Option 1: From Project Root
```bash
streamlit run dashboard/app.py
```

### Option 2: Specify Full Path
```bash
cd d:\Documents\sem-8\Major Project phase 2\Implement
streamlit run dashboard/app.py
```

### Option 3: Remote Access
```bash
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

Then access at: `http://localhost:8501`

## Dashboard Layout

### Sidebar
- **Live Auto-Refresh** toggle (updates every 5 seconds)
- **Refresh Interval** selector
- **System Status** indicator (shows if monitor is active)
- **About** information

### Main Content

**Top Row - KPIs**
- 4 metric cards showing key statistics
- Threats highlighted in red if count > 0

**Middle Row - Analytics**
- Domain distribution pie chart
- Attachment types bar chart

**Bottom Row - Logs & Threats**
- Three tabs for different views:
  - **Threats Only**: Shows only `[THREAT]` entries
  - **Recent Activity**: Last 50 log lines with highlighting
  - **Full Log**: Raw log content

**Footer**
- File status indicators
- Last update timestamp

## Data Files

The dashboard reads from the project root:

```
d:\Documents\sem-8\Major Project phase 2\Implement\
├── .sentinel_metrics.json          ← Metrics (total analyzed, threats, etc.)
├── .sentinel_shield.log            ← Live monitoring logs
├── .sentinel_threat_registry.json  ← Persistent threat history
└── dashboard/
    ├── app.py                       ← Dashboard script
    └── requirements.txt             ← Python dependencies
```

## File Updates

The dashboard uses caching with 5-second TTL for performance:
- Metrics are cached for 5 seconds
- Logs are cached for 5 seconds
- Live auto-refresh forces new data fetching

## Troubleshooting

### Dashboard Shows No Data
1. Ensure monitor is running: `python scripts/inference/9_gmail_live_monitor.py`
2. Check if files exist in project root:
   ```bash
   ls -la ../.sentinel_*.json ../.sentinel_*.log
   ```
3. Wait 5-10 seconds for monitor to create initial files

### Files Not Found Errors
- Dashboard provides sensible defaults (0 values)
- Displays warning messages if files can't be read
- Shows "No logs found" if monitor hasn't run yet

### Slow Performance
- Adjust the 5-second cache timeout in `app.py`
- Disable auto-refresh if not needed
- Dashboard caches data aggressively for performance

### Port Already in Use
```bash
streamlit run dashboard/app.py --server.port 8502
# Use port 8502 instead of default 8501
```

## Customization

### Change Refresh Rate
Edit `app.py` line ~95:
```python
@st.cache_data(ttl=5)  # Change 5 to desired seconds
```

### Change Chart Colors
Edit `app.py` in the visualization sections to modify colors and themes.

### Add More Metrics
1. Add new column in KPI section
2. Extract from `metrics` dict
3. Use `st.metric()` to display

### Customize Log Highlighting
Edit the `highlight_threats()` function in `app.py` to change which keywords trigger highlighting.

## Performance Notes

- Dashboard handles missing files gracefully
- Default values prevent crashes if metrics aren't ready
- Caching reduces file I/O overhead
- Threat registry loads efficiently even with 100+ threats

## Integration with Monitor

The dashboard automatically picks up data from the monitor:
1. Monitor writes to `.sentinel_metrics.json` every analysis
2. Monitor appends to `.sentinel_shield.log` continuously
3. Monitor records threats in `.sentinel_threat_registry.json`
4. Dashboard reads these files with 5-second refresh

**Recommended Setup:**
- Run monitor in one terminal: `python scripts/inference/9_gmail_live_monitor.py`
- Run dashboard in another terminal: `streamlit run dashboard/app.py`
- Access dashboard at `http://localhost:8501`

## 📂 Directory & Files

### Dashboard Files Explained

| File | Purpose | Edit? |
|------|---------|-------|
| `app.py` | Main Streamlit application (500+ lines) | No |
| `config.json` | All dashboard settings | Yes |
| `requirements.txt` | Python dependencies | No |
| `README.md` | This file - complete documentation | No |
| `QUICK_START.md` | 5-minute getting started guide | No |

### Configuration Options

Edit `dashboard/config.json` to customize:

```json
{
  "refresh_interval_seconds": 5,          // Change refresh rate
  "logs": {
    "max_lines_to_display": 50            // Change log count
  },
  "threat_registry": {
    "max_threats_to_display": 10,         // Change threat count
    "confidence_high_threshold": 0.8      // Change color thresholds
  },
  "colors": {
    "threat_background": "#ffcccc",       // Customize colors
    "threat_text": "#cc0000"
  }
}
```

### Data Integration

Dashboard reads from project root:
```
├── .sentinel_metrics.json           (Updated per analysis)
├── .sentinel_shield.log             (Appended continuously)  
└── .sentinel_threat_registry.json   (Persistent, never resets)
```

These files are created and updated by the monitor script (`9_gmail_live_monitor.py`).

## Version History

- **v1.0** (April 19, 2026)
  - Initial release
  - KPI metrics display
  - Domain and attachment visualizations
  - Live threat feed
  - Auto-refresh capability
  - Persistent threat registry integration

---

**Questions or Issues?** Check the main project README or monitor logs.
