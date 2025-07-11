from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.base import BaseConfig
from .routes import router, initialize_optimizer
from .models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "api.log")
    ]
)

logger = logging.getLogger(__name__)

def create_app(config: BaseConfig = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    if config is None:
        config = BaseConfig()
    
    # Create FastAPI app
    app = FastAPI(
        title="DLD Optimization API",
        description="API for optimizing DLD (Deterministic Lateral Displacement) geometry parameters",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        try:
            # Ensure logs directory exists
            config.LOGS_DIR.mkdir(exist_ok=True)
            
            # Initialize optimizer
            initialize_optimizer(config)
            logger.info("Application startup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Application shutting down")
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                details=str(exc)
            ).dict()
        )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "DLD Optimization API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    return app

# Create app instance for direct execution
app = create_app()

if __name__ == "__main__":
    import uvicorn
    from config.base import BaseConfig
    
    config = BaseConfig()
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        workers=config.API_WORKERS if not config.DEBUG else 1
    ) 