#!/bin/bash

# Stop Script for ATC Voice Live System
# Stops all running ATC Voice processes

echo "🛑 Stopping ATC Voice Live System"
echo "================================="

# Stop logext.py (PRESERVED - don't stop it)
echo "🎙️ Preserving communications detection (logext.py)..."
echo "ℹ️ logext.py will continue running as requested"

# Stop dashboard
echo "🌐 Stopping dashboard..."
pkill -f "streamlit.*app.py" 2>/dev/null && echo "✅ Dashboard stopped" || echo "ℹ️ Dashboard not running"

# Stop any other ATC Voice processes
echo "🧹 Cleaning up other processes..."
pkill -f "python.*all_in_one.py" 2>/dev/null && echo "✅ Audio recording stopped" || echo "ℹ️ Audio recording not running"
pkill -f "python.*atlas.py.*--live" 2>/dev/null && echo "✅ Live postprocessor stopped" || echo "ℹ️ Live postprocessor not running"

# Kill any processes using port 8501
echo "🔌 Freeing port 8501..."
lsof -ti :8501 | xargs kill -9 2>/dev/null && echo "✅ Port 8501 freed" || echo "ℹ️ Port 8501 was already free"

# Wait a moment
sleep 2

# Verify processes are stopped (except logext.py)
echo ""
echo "🔍 Checking remaining processes..."
REMAINING=$(ps aux | grep -E "(streamlit|all_in_one|atlas.py.*--live)" | grep -v grep | wc -l)

if [ "$REMAINING" -eq 0 ]; then
    echo "✅ All ATC Voice processes stopped successfully (except logext.py)"
else
    echo "⚠️ Some processes may still be running:"
    ps aux | grep -E "(streamlit|all_in_one|atlas.py.*--live)" | grep -v grep
fi

# Check if logext.py is still running (it should be)
if ps aux | grep -q "python.*logext.py" | grep -v grep; then
    LOGEXT_PID=$(ps aux | grep "python.*logext.py" | grep -v grep | awk '{print $2}' | head -1)
    echo "✅ logext.py preserved and still running (PID: $LOGEXT_PID)"
else
    echo "⚠️ logext.py is not running"
fi

echo ""
echo "🛑 ATC Voice Live System stopped (logext.py preserved)"
