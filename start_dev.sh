#!/bin/bash

echo "🚀 Starting DLD Optimization Services (Local Development)..."
echo ""

# Stop any existing processes
pkill -f "uvicorn.*backend.api.app" 2>/dev/null
pkill -f "streamlit.*app/main.py" 2>/dev/null
sleep 1

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    pkill -f "uvicorn.*backend.api.app" 2>/dev/null
    pkill -f "streamlit.*app/main.py" 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
echo "🔧 Starting backend API..."
python3 -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload &
sleep 2

# Start frontend (foreground - script will wait here)
echo "🎨 Starting Streamlit frontend..."
echo "📱 Frontend: http://localhost:8501 | Backend: http://localhost:8000/docs"
echo "🛑 Press Ctrl+C to stop"
echo ""
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0

