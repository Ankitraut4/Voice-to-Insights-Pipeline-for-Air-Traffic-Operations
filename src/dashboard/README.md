# ATC Voice Communications Dashboard - Live

A comprehensive Streamlit dashboard for visualizing and analyzing live Air Traffic Control voice communications data.

## Features

### 📊 Communication Categorization
- **Frequency Handoffs**: Contact approach, switch to tower frequency, handoff to center
- **Heading Vectors**: Turn left/right, fly heading, maintain present heading  
- **Altitude Clearances**: Descend/climb and maintain, expedite climb
- **Emergency Declarations**: Mayday, pan pan pan, emergency landing requests
- **Miscellaneous**: Roger, weather updates, runway assignments, takeoff clearances

### 📈 Visualization & Analytics
- **Daily Communication Counts**: Total voice communications per day
- **Category Breakdown**: Daily counts by communication category
- **Aggregate Statistics**: Min, median, mean, average, and standard deviation
- **Histograms**: Communication frequency and duration distributions

### 🔍 Pattern Analysis
- **Communications per Flight**: Number of transmissions per aircraft
- **Time Intervals**: Analysis of time between transmissions
- **Hourly Patterns**: Activity throughout the day (higher during business hours)
- **Day of Week Patterns**: Weekly communication trends
- **Redundant Transmissions**: Detection of repeated communications

## Quick Start

1. **Run the dashboard** (automatically sets up environment):
   ```bash
   ./run_dashboard.sh
   ```

2. **Or manually**:
   ```bash
   # Activate virtual environment
   source venv/bin/activate
   
   # Install dependencies if needed
   pip install streamlit pandas numpy plotly
   
   # Run the dashboard
   streamlit run src/dashboard/app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## Live Data Sources

The dashboard reads from:
- **Transcriptions**: `src/data/logs/transcripts/categorized_transcription_results.json`
- **Communications**: `src/data/logs/atc_communications.txt`

Both files are runtime-generated and may be absent before the first pipeline run.

## Dashboard Features

- **Manual Refresh**: Use the sidebar **Refresh Data Now** button
- **Browser Refresh**: You can also refresh the page with browser refresh (for example, F5)
- **Real Data**: Uses actual transcription and communication log data
- **Interactive Tabs**: Switch between different analysis views

## Data Structure

The dashboard processes real ATC data:
- **Transcriptions**: Categorized communications with timestamps, flight numbers, and duration
- **Communications**: Live audio detection logs with dBFS levels
- **Flight Numbers**: Extracted from transcription text using pattern matching
- **Categories**: Pre-categorized communications (Frequency Handoffs, Heading Vectors, etc.)
- **Timestamps**: Real UTC timestamps from the data sources

## Technical Details

- **Framework**: Streamlit with Plotly for interactive charts
- **Data Processing**: Pandas for analysis, NumPy for calculations
- **Caching**: Uses `@st.cache_data` and `@st.cache_resource`
- **Current cache TTLs in `src/dashboard/app.py`**:
  - Core file loaders use short TTLs (`ttl=3`)
  - Some computed views/statistics use longer TTLs (`ttl=10` or `ttl=60`)
- **Responsive**: Wide layout optimized for desktop viewing

## Future Enhancements

- Integration with real ATC data sources
- Real-time streaming updates
- Export functionality for reports
- Advanced NLP analysis
- Alert system for anomalies
