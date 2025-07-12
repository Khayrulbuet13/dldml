#!/bin/bash

# Activate virtual environment
source ~/.virtualenvs/dld/bin/activate

# Kill any existing processes
echo "Stopping any existing services..."
pkill -f "uvicorn.*backend.api.app" 2>/dev/null
pkill -f "streamlit.*app/main.py" 2>/dev/null
sleep 2

# Start backend in background
echo "Starting backend API..."
uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "Starting Streamlit frontend..."
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0

# Cleanup function
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    pkill -f "uvicorn.*backend.api.app" 2>/dev/null
    pkill -f "streamlit.*app/main.py" 2>/dev/null
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Wait for background process
wait $BACKEND_PID