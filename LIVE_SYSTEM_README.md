# ATC Voice Live Runtime (Canonical)

## Overview
This repository's canonical live runtime is:
- `src/utils/all_in_one.py` for audio ingestion and Whisper transcription
- `src/nlp_analysis/atlas.py --live` as the single live NLP processor
- `src/dashboard/app.py` for the Streamlit dashboard

`postprocess.py` is not the primary live processor in this runtime path.

## Canonical Live Flow
1. `all_in_one.py` records audio chunks and writes transcript entries to:
   - `src/data/logs/transcripts/transcripts.json`
2. `atlas.py --live` monitors transcript updates and writes categorized output to:
   - `src/data/logs/transcripts/categorized_transcription_results.json`
3. Streamlit reads categorized transcripts and communications logs for visualization.

Optional parallel ingestion utility:
- `src/data_ingestion/logext.py` writes communications events to:
  - `src/data/logs/atc_communications.txt`

## Entrypoints

### Linux shell entrypoints
```bash
./run_live_system.sh
./start_live.sh
./stop_live.sh
```

### Python NLP entrypoints
```bash
python src/nlp_analysis/atlas.py
python src/nlp_analysis/atlas.py --live
```

`run_audio_recording.sh` and `run_live_postprocessing.sh` are not current entrypoints in this repository.

## Runtime Data Files
- `src/data/logs/transcripts/transcripts.json`
  - Produced by ingestion/transcription (`all_in_one.py`)
  - May be created at runtime if absent before first run
- `src/data/logs/transcripts/categorized_transcription_results.json`
  - Produced/updated by NLP processing (`atlas.py` or `atlas.py --live`)
  - May be created at runtime if absent before first run
- `src/data/logs/atc_communications.txt`
  - Produced by communications detection (`logext.py`)
  - Runtime-generated and may be absent before first run

## Linux-Only Notes
- `run_live_system.sh`, `start_live.sh`, `stop_live.sh`, and systemd setup scripts are Linux-oriented shell flows.
- If using systemd (`install_as_service.sh`), run that only on Linux hosts with systemd available.
