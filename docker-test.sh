#!/bin/bash

# Test script for Docker Compose setup

echo "🔍 Testing Docker Compose configuration..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Check if required files exist
required_files=("docker-compose.yml" "Dockerfile.backend" "Dockerfile.frontend" "requirements.txt" "nginx.conf")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file $file is missing"
        exit 1
    fi
    echo "✅ Found $file"
done

# Check if required directories exist
required_dirs=("backend" "app" "config")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Required directory $dir is missing"
        exit 1
    fi
    echo "✅ Found directory $dir"
done

echo ""
echo "🚀 Starting Docker Compose services..."

# Build and start services
docker-compose up --build -d

echo ""
echo "⏳ Waiting for services to start..."

# Wait for services to be healthy
sleep 30

# Check service health
echo ""
echo "🏥 Checking service health..."

# Check backend health
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
fi

# Check frontend health
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend health check failed"
fi

echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🌐 Access URLs:"
echo "Frontend: http://localhost:8501"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/api/v1/health"

echo ""
echo "📝 To stop services, run: docker-compose down"
echo "📝 To view logs, run: docker-compose logs -f" 