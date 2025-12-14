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

# Function to get available disk space in GB
get_available_space() {
    df -BG / 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//' || echo "0"
}

# Check available disk space (for 4GB constraint optimization)
AVAILABLE_SPACE=$(get_available_space)
echo "💾 Available disk space: ${AVAILABLE_SPACE}GB"
echo ""

# If space is limited (< 10GB), use space-efficient build strategy
if [ "$AVAILABLE_SPACE" -lt 10 ] 2>/dev/null; then
    echo "⚠️  Low disk space detected. Using space-efficient build strategy..."
    echo ""
    
    # Clean build cache first
    echo "🧹 Step 1/5: Cleaning Docker build cache..."
    docker builder prune -af --filter "until=24h" >/dev/null 2>&1 || true
    echo "   ✓ Build cache cleaned"
    echo ""
    
    # Remove old images if they exist
    echo "🗑️  Step 2/5: Removing old images (if any)..."
    docker rmi -f dldml-backend:latest dldml-frontend:latest 2>/dev/null || true
    docker system prune -f >/dev/null 2>&1 || true
    echo "   ✓ Old images removed"
    echo ""
    
    # Build backend first
    echo "🔨 Step 3/5: Building backend image..."
    docker-compose build --no-cache backend
    if [ $? -ne 0 ]; then
        echo "❌ Backend build failed!"
        exit 1
    fi
    echo "   ✓ Backend built successfully"
    echo ""
    
    # Clean build cache after backend
    echo "🧹 Step 4/5: Cleaning build cache after backend..."
    docker builder prune -af --filter "until=1h" >/dev/null 2>&1 || true
    echo "   ✓ Cache cleaned"
    echo ""
    
    # Build frontend
    echo "🔨 Step 5/5: Building frontend image..."
    docker-compose build --no-cache frontend
    if [ $? -ne 0 ]; then
        echo "❌ Frontend build failed!"
        exit 1
    fi
    echo "   ✓ Frontend built successfully"
    echo ""
    
    # Final cleanup
    echo "🧹 Final cleanup..."
    docker builder prune -af >/dev/null 2>&1 || true
    echo "   ✓ Cleanup complete"
    echo ""
    
    # Start containers
    echo "🚀 Starting containers..."
    docker-compose up -d
else
    # Normal build for systems with sufficient space
    echo "📦 Building and starting containers (normal mode)..."
    docker-compose up -d --build
fi

echo ""
echo "✅ Services started in Docker!"
echo ""
echo "📊 Image sizes:"
docker images --format "   {{.Repository}}:{{.Tag}} - {{.Size}}" | grep "dldml" || echo "   (checking...)"

# Show disk usage
CURRENT_SPACE=$(get_available_space)
echo ""
echo "💾 Current disk space: ${CURRENT_SPACE}GB"
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

