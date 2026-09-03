#!/bin/bash

# Quick Start Script for ATC Voice Live System
# Delegates to run_live_system.sh for canonical orchestration

echo "🚀 Starting ATC Voice Live System (Unified All-in-One)"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "src/nlp_analysis/atlas.py" ]; then
    echo "❌ Error: Please run this script from the ATC-Voice root directory"
    exit 1
fi

# Verify canonical launcher exists
if [ ! -f "./run_live_system.sh" ]; then
    echo "❌ Error: run_live_system.sh not found"
    exit 1
fi

echo "✅ Canonical flow: all_in_one.py ingestion/transcription + atlas.py --live processing"
echo "🔁 Delegating startup to run_live_system.sh"

exec bash ./run_live_system.sh
