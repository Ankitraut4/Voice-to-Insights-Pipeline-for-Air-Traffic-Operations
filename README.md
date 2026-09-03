# Voice-to-Insights Pipeline for Air Traffic Operations

## 1) Project Overview

This repository is a modular prototype for turning ATC voice/transcription streams into structured, searchable operational insights.

The focus is practical engineering: ingest communications, normalize noisy transcripts, detect callsigns, classify message intent, and surface results in an interactive analytics dashboard.

This is a pipeline-style implementation, not a production-scale distributed platform.

## 2) Processing Flow

1. **Audio / communication ingestion**
   - `src/data_ingestion/logext.py` monitors a LiveATC stream and detects communication activity from audio levels.
   - Runtime communication detections are appended to `src/data/logs/atc_communications.txt` (generated at runtime, git-ignored, and potentially absent before first run).
2. **Transcript normalization**
   - `src/nlp_analysis/atlas.py` and `src/nlp_analysis/postprocess.py` normalize punctuation/noise and convert ATC spoken numbers.
3. **Callsign detection**
   - Detects commercial aliases plus general-aviation N-numbers (direct and "november ..." forms).
4. **Communication categorization**
   - Rule-based categorization using canonical keyword/callsign config.
   - Includes emergency false-positive gating for ambiguous "emergency" references.
5. **Structured JSON output**
   - Writes categorized transcript records to `src/data/logs/transcripts/categorized_transcription_results.json` (generated at runtime and potentially absent before first run).
   - Supports incremental append behavior and atomic temp-file replacement in processing code.
6. **Streamlit analytics dashboard**
   - `src/dashboard/app.py` reads categorized transcripts + communication logs and renders operational metrics/visualizations.

## 3) Key Engineering Features

- Bounded audio buffering in stream ingestion.
- Silence/communication detection via dBFS thresholding.
- Reconnection with exponential backoff for stream errors.
- ATC-specific spoken-number normalization.
- Commercial + general-aviation callsign detection.
- Emergency-context gating to reduce false-positive emergency labels.
- Incremental transcript processing with duplicate tracking and atomic output writes.
- Portable project-root resolution in core scripts with `ATC_VOICE_ROOT` override.
- Automated pytest coverage for deterministic normalization/callsign/categorization behaviors.

## 4) Actual Repository Structure

```text
.
├── README.md
├── LIVE_SYSTEM_README.md
├── requirements.txt
├── requirements-dev.txt
├── run_dashboard.sh
├── run_live_system.sh
├── run_postprocessing_pipeline.py
├── install_as_service.sh
├── final_clean.py
├── repair_json.py
├── live_postprocessor.py
├── start_live.sh
├── stop_live.sh
├── stop_atlas_and_cleaner.sh
├── config/
│   ├── final_aviation_ultimate_with_emergency.json
│   ├── category_dict.json
│   ├── airline_callsign.json
│   ├── phonetic_alphabet.json
│   ├── airline_nnumbers.json
├── src/
│   ├── data_ingestion/
│   │   └── logext.py
│   ├── nlp_analysis/
│   │   ├── atlas.py
│   │   ├── postprocess.py
│   │   └── auto_cleaner.py
│   ├── dashboard/
│   │   ├── app.py
│   │   └── README.md
│   └── utils/
│       ├── all_in_one.py
│       ├── airline_count.py
│       ├── pair_timeinterval_atc.py
│       ├── map.jpeg
│       ├── ZNYHighAltitudeCharts.jpg
└── tests/
    ├── test_transcript_processing.py
    ├── test_callsign_detection.py
    └── test_categorization.py
```

## 5) Installation

```bash
python -m venv venv
```

Linux/macOS:
```bash
source venv/bin/activate
```

Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Optional developer dependencies:

```bash
pip install -r requirements-dev.txt
```

## 6) Running

### Dashboard only

Linux-oriented helper script:
```bash
./run_dashboard.sh
```

Direct Python entrypoint:
```bash
python -m streamlit run src/dashboard/app.py
```

### Postprocessing pipeline + dashboard launcher

```bash
python run_postprocessing_pipeline.py
```

This runs transcript categorization and then starts Streamlit.

### NLP categorization engine

One-time pass:
```bash
python src/nlp_analysis/atlas.py
```

Live monitoring mode:
```bash
python src/nlp_analysis/atlas.py --live
```

### Linux live-system orchestration scripts

```bash
./run_live_system.sh
./start_live.sh
./stop_live.sh
./install_as_service.sh
```

These scripts are Linux/systemd-oriented and rely on tools such as `bash`, `pkill`, `lsof`, and `systemctl`.

## 7) Testing

Run tests:

```bash
python -m pytest -v
```

Current focused suite: **13 tests** across transcript processing, callsign detection, and categorization.

## 8) Configuration

Canonical configuration files in `config/` used by current processing/dashboard code:

- `config/final_aviation_ultimate_with_emergency.json`
- `config/category_dict.json`
- `config/airline_callsign.json`
- `config/phonetic_alphabet.json`
- `config/airline_nnumbers.json`

Portable path resolution:

- Core scripts support `ATC_VOICE_ROOT` to override project-root discovery.
- If unset, scripts fall back to deriving the root from script location.

Runtime artifacts:

- Runtime-generated logs/data are intentionally ignored by Git via `.gitignore` rules (for example `logs/`, `*.log`, `src/data/logs/` paths).

## 9) Limitations

- Categorization is primarily rule-based and regex-driven.
- Persistence is file/JSON-based rather than database/event-backed.
- End-to-end behavior depends on external audio/transcription tooling and stream availability.
- Dashboard and processing logic still have partial coupling through shared data files and duplicated helper logic.
- Linux orchestration/service scripts are not portable to all environments without adaptation.

## 10) Future Improvements

- Move from JSON files to database/event-backed persistence.
- Consolidate shared domain logic (normalization, callsign parsing, categorization helpers) into reusable modules.
- Add broader integration and pipeline-level tests beyond the current focused unit suite.
- Improve deployment ergonomics with stronger containerization/runtime profiles.

