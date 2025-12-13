#!/bin/bash

echo "🐳 Starting DLD Optimization Services in Docker..."
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker and docker-compose."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start docker-compose
echo "📦 Building and starting containers..."
docker-compose up -d --build

echo ""
echo "✅ Services started in Docker!"
echo ""
echo "📱 Access the application:"
echo "   Frontend (Streamlit): http://localhost:8501"
echo "   Backend API:          http://localhost:8000"
echo "   API Documentation:    http://localhost:8000/docs"
echo ""
echo "📊 View logs: docker-compose logs -f"
echo "📊 View backend logs: docker-compose logs -f backend"
echo "📊 View frontend logs: docker-compose logs -f frontend"
echo "🛑 Stop services: docker-compose down"
echo "🔄 Restart services: docker-compose restart"
echo ""

