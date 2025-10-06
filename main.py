from fastapi import FastAPI, HTTPException, Query, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os
from pathlib import Path
import time
from typing import List, Optional, Dict
import traceback
from pydantic import BaseModel, Field
import math
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import modules with fallback
try:
    from hybrid_recommender import HybridRecommender
    from utils import (
        batch_format_recommendations,
        PerformanceMonitor,
        RecommendationCache
    )
    import config
except ImportError as e:
    print(f"Import error: {e}")
    class Config:
        API_HOST = '0.0.0.0'
        API_PORT = 8000
        DEBUG = True
        CACHE_SIZE = 2000
        CACHE_TTL = 7200
        DEFAULT_RECOMMENDATIONS = 20
    config = Config()

# --- Utility Functions ---
def safe_convert(value, converter, default):
    """Safely convert values with fallback"""
    if value is None or value == "":
        return default
    try:
        converted = converter(value)
        if converter == float and (math.isnan(converted) or math.isinf(converted)):
            return default
        return converted
    except (ValueError, TypeError):
        return default

# --- Pydantic Models ---
class MovieResponse(BaseModel):
    id: int
    title: str
    year: Optional[int] = None
    rating: float = 0.0
    genre_names: List[str] = []
    overview: str = ""
    runtime: Optional[int] = None
    director: str = ""
    similarity_score: float = 0.0
    recommendation_reason: str = "Recommended"
    explanation: str = ""
    poster_url: Optional[str] = None
    
    class Config:
        extra = "allow"

class UserRatingRequest(BaseModel):
    movie_id: int = Field(..., ge=1)
    rating: float = Field(..., ge=0.5, le=5.0)

class OnboardingRequest(BaseModel):
    liked_movie_ids: List[int] = Field(..., description="Movie IDs liked during onboarding")

class RecommendationResponse(BaseModel):
    recommendations: List[MovieResponse]
    total_count: int
    recommendation_type: str
    generated_at: float
    user_type: Optional[str] = None

# --- FastAPI App ---
app = FastAPI(
    title="FilmFusion - Enhanced Hybrid Recommender API",
    description="AI-powered movie recommendation system with improved collaborative filtering",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
try:
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
    templates = Jinja2Templates(directory="frontend/templates")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")
    templates = None

# Global instances
recommender: Optional[HybridRecommender] = None
performance_monitor = PerformanceMonitor()
recommendation_cache = RecommendationCache(
    max_size=config.CACHE_SIZE, 
    ttl=config.CACHE_TTL
)
user_ids_cache: List[int] = []

# --- Helper Functions ---
def load_user_ids():
    """Load available user IDs for autocomplete"""
    global user_ids_cache
    try:
        ratings_df = pd.read_csv(config.FILES['ratings_clean'])
        if 'userId' in ratings_df.columns:
            user_ids_cache = sorted(ratings_df['userId'].unique().tolist())
            print(f"Loaded {len(user_ids_cache)} unique user IDs")
    except Exception as e:
        print(f"Could not load user IDs: {e}")
        user_ids_cache = []

def determine_user_type(user_id: int) -> str:
    """Determine user type based on rating history"""
    if not recommender:
        return "unknown"
    
    try:
        # Check collaborative recommender first
        if recommender.collaborative_recommender.loaded:
            user_profile = recommender.collaborative_recommender.get_user_profile(user_id)
            if user_profile:
                total_ratings = user_profile.get('total_ratings', 0)
            else:
                total_ratings = 0
        else:
            total_ratings = 0
        
        # Check in-memory ratings
        from utils import get_user_ratings
        user_ratings = get_user_ratings(user_id)
        if user_ratings:
            total_ratings += len(user_ratings)
        
        # Classify user type based on updated thresholds
        if total_ratings >= config.COLLABORATIVE_THRESHOLD_EXPERT:
            return "power_user"
        elif total_ratings >= config.COLLABORATIVE_THRESHOLD_HIGH:
            return "active_user" 
        elif total_ratings >= config.COLLABORATIVE_THRESHOLD_MEDIUM:
            return "moderate_user"
        elif total_ratings >= config.COLLABORATIVE_THRESHOLD_LOW:
            return "light_user"
        else:
            return "new_user"
            
    except Exception as e:
        print(f"Error determining user type: {e}")
        return "unknown"

# --- Startup and Dependencies ---
@app.on_event("startup")
async def startup_event():
    global recommender
    print("Starting FilmFusion Enhanced API...")
    
    # Validate configuration
    if hasattr(config, 'validate_config'):
        config.validate_config()
    
    # Load system components
    load_user_ids()
    recommender = HybridRecommender()
    recommender.load_models()
    
    print("Enhanced recommendation system loaded successfully!")
    print(f"- Content weight: {recommender.base_content_weight}")
    print(f"- Collaborative weight: {recommender.base_collaborative_weight}")
    print(f"- Collaborative threshold lowered to: {config.COLLABORATIVE_THRESHOLD_LOW}")

def get_recommender() -> HybridRecommender:
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation system not available")
    return recommender

# --- Core API Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("<h1>FilmFusion API</h1><p>Frontend not available. Use API endpoints.</p>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy" if recommender and recommender.loaded else "initializing",
        "version": "2.1.0",
        "features_enabled": config.FEATURES if hasattr(config, 'FEATURES') else {},
        "collaborative_threshold": config.COLLABORATIVE_THRESHOLD_LOW
    }

# --- User Management Routes ---
@app.get("/api/users/{user_id}/status")
async def get_user_status(user_id: int, rec: HybridRecommender = Depends(get_recommender)):
    """Get comprehensive user status including type and recommendation readiness"""
    try:
        user_type = determine_user_type(user_id)
        is_new = rec.is_new_user(user_id)
        
        # Get user profile if available
        user_profile = None
        try:
            if rec.collaborative_recommender.loaded:
                user_profile = rec.collaborative_recommender.get_user_profile(user_id)
        except:
            pass
        
        return {
            "user_id": user_id,
            "user_type": user_type,
            "is_new_user": is_new,
            "needs_onboarding": is_new,
            "profile": user_profile,
            "recommendation_ready": not is_new
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/onboarding-movies")
async def get_onboarding_movies(user_id: int, rec: HybridRecommender = Depends(get_recommender)):
    """Get diverse movies for user onboarding"""
    try:
        movies = rec.get_cold_start_movies()
        formatted_movies = batch_format_recommendations(movies)
        
        return {
            "user_id": user_id,
            "movies": formatted_movies,
            "instruction": "Rate at least 3 movies to get personalized recommendations"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/{user_id}/complete-onboarding", response_model=RecommendationResponse)
async def complete_onboarding(
    user_id: int, 
    request: OnboardingRequest, 
    rec: HybridRecommender = Depends(get_recommender)
):
    """Generate initial recommendations after onboarding"""
    try:
        recommendations = rec.generate_onboarding_recommendations(
            user_id, 
            request.liked_movie_ids
        )
        formatted_recs = batch_format_recommendations(recommendations)
        
        # Clear cache for this user
        recommendation_cache.clear_user(user_id)
        
        return RecommendationResponse(
            recommendations=formatted_recs,
            total_count=len(formatted_recs),
            recommendation_type="onboarding_personalized",
            generated_at=time.time(),
            user_type="new_user"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to generate onboarding recommendations")

@app.post("/api/users/{user_id}/rate")
async def rate_movie(
    user_id: int, 
    request: UserRatingRequest, 
    rec: HybridRecommender = Depends(get_recommender)
):
    """Rate a movie and update user preferences"""
    try:
        rec.update_user_preferences(user_id, request.movie_id, request.rating)
        
        # Clear caches to ensure fresh recommendations
        recommendation_cache.clear_user(user_id)
        
        # Determine new user type after rating
        user_type = determine_user_type(user_id)
        
        return {
            "message": "Rating recorded successfully",
            "user_type": user_type,
            "rating": request.rating,
            "movie_id": request.movie_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error recording rating")

@app.get("/api/users/search")
async def search_users(
    query: str = Query("", min_length=1), 
    limit: int = Query(10, ge=1, le=50)
):
    """Search for user IDs"""
    if not user_ids_cache:
        return {"results": []}
    
    try:
        query_str = str(query)
        matching_users = [
            uid for uid in user_ids_cache 
            if query_str in str(uid)
        ][:limit]
        
        return {"results": matching_users}
    except Exception as e:
        return {"results": []}

# --- Movie and Search Routes ---
@app.get("/api/movies/search", response_model=List[MovieResponse])
async def search_movies(
    query: str = Query(..., min_length=1), 
    limit: int = Query(20, ge=1, le=50), 
    rec: HybridRecommender = Depends(get_recommender)
):
    """Enhanced movie search"""
    try:
        # Check cache first
        cache_key = f"search_{query}_{limit}"
        cached_result = recommendation_cache.get(cache_key)
        if cached_result:
            return cached_result
            
        results = rec.content_recommender.search_movies(query, limit)
        formatted_results = batch_format_recommendations(results)
        
        # Cache results
        recommendation_cache.set(cache_key, formatted_results)
        
        return formatted_results
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/api/movies/{movie_id}/details", response_model=MovieResponse)
async def get_movie_details(
    movie_id: int, 
    rec: HybridRecommender = Depends(get_recommender)
):
    """Get detailed movie information"""
    try:
        details = rec.content_recommender.get_movie_details(movie_id)
        if not details:
            raise HTTPException(status_code=404, detail="Movie not found")
            
        formatted_details = batch_format_recommendations([details])
        return formatted_details[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving movie details")

@app.get("/api/movies/{movie_id}/similar", response_model=RecommendationResponse)
async def get_similar_movies(
    movie_id: int, 
    limit: int = Query(20, ge=1, le=50), 
    rec: HybridRecommender = Depends(get_recommender)
):
    """Get similar movies using hybrid approach"""
    try:
        # Check cache
        cache_key = f"similar_{movie_id}_{limit}"
        cached_result = recommendation_cache.get(cache_key)
        if cached_result:
            return RecommendationResponse(
                recommendations=cached_result,
                total_count=len(cached_result),
                recommendation_type="similar",
                generated_at=time.time()
            )
        
        recommendations = rec.get_similar_movies(movie_id, n_recommendations=limit)
        formatted_recs = batch_format_recommendations(recommendations)
        
        # Cache results
        recommendation_cache.set(cache_key, formatted_recs)
        
        return RecommendationResponse(
            recommendations=formatted_recs,
            total_count=len(formatted_recs),
            recommendation_type="similar",
            generated_at=time.time()
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error getting similar movies")

# --- Main Recommendation Routes ---
@app.get("/api/recommendations/personalized/{user_id}", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    user_id: int, 
    limit: int = Query(20, ge=1, le=50),
    force_refresh: bool = Query(False, description="Force refresh cache"),
    rec: HybridRecommender = Depends(get_recommender)
):
    """Get personalized recommendations with improved caching"""
    start_time = time.time()
    
    try:
        # Check cache unless force refresh
        if not force_refresh:
            cache_key = f"personalized_{user_id}_{limit}"
            cached_result = recommendation_cache.get(cache_key)
            if cached_result:
                return RecommendationResponse(
                    recommendations=cached_result,
                    total_count=len(cached_result),
                    recommendation_type="personalized_cached",
                    generated_at=time.time(),
                    user_type=determine_user_type(user_id)
                )
        
        # Generate fresh recommendations
        recommendations = rec.get_personalized_recommendations(
            user_id, 
            n_recommendations=limit,
            include_explanations=True
        )
        
        formatted_recs = batch_format_recommendations(recommendations)
        user_type = determine_user_type(user_id)
        
        # Cache results
        cache_key = f"personalized_{user_id}_{limit}"
        recommendation_cache.set(cache_key, formatted_recs)
        
        # Log performance
        performance_monitor.log_recommendation_time(time.time() - start_time)
        
        return RecommendationResponse(
            recommendations=formatted_recs,
            total_count=len(formatted_recs),
            recommendation_type="personalized",
            generated_at=time.time(),
            user_type=user_type
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error generating personalized recommendations")

@app.get("/api/recommendations/popular")
async def get_popular_recommendations(
    limit: int = Query(20, ge=1, le=50),
    rec: HybridRecommender = Depends(get_recommender)
):
    """Get popular/trending movies"""
    try:
        # Check cache
        cache_key = f"popular_{limit}"
        cached_result = recommendation_cache.get(cache_key)
        if cached_result:
            return RecommendationResponse(
                recommendations=cached_result,
                total_count=len(cached_result),
                recommendation_type="popular_cached",
                generated_at=time.time()
            )
        
        recommendations = rec.content_recommender.get_top_rated_movies(limit)
        formatted_recs = batch_format_recommendations(recommendations)
        
        # Cache for longer time (popular movies don't change often)
        recommendation_cache.set(cache_key, formatted_recs, ttl=config.CACHE_TTL * 2)
        
        return RecommendationResponse(
            recommendations=formatted_recs,
            total_count=len(formatted_recs),
            recommendation_type="popular",
            generated_at=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error getting popular recommendations")

@app.get("/api/recommendations/by-genre/{genre}")
async def get_genre_recommendations(
    genre: str,
    limit: int = Query(20, ge=1, le=50),
    rec: HybridRecommender = Depends(get_recommender)
):
    """Get recommendations by genre"""
    try:
        # Check cache
        cache_key = f"genre_{genre.lower()}_{limit}"
        cached_result = recommendation_cache.get(cache_key)
        if cached_result:
            return RecommendationResponse(
                recommendations=cached_result,
                total_count=len(cached_result),
                recommendation_type=f"{genre}_cached",
                generated_at=time.time()
            )
        
        recommendations = rec.content_recommender.get_movies_by_genre(genre, limit)
        formatted_recs = batch_format_recommendations(recommendations)
        
        # Cache results
        recommendation_cache.set(cache_key, formatted_recs)
        
        return RecommendationResponse(
            recommendations=formatted_recs,
            total_count=len(formatted_recs),
            recommendation_type=f"{genre}_genre",
            generated_at=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting {genre} recommendations")

# --- Analytics and Admin Routes ---
@app.get("/api/analytics/system-stats")
async def get_system_statistics(rec: HybridRecommender = Depends(get_recommender)):
    """Get comprehensive system analytics"""
    try:
        # Get model statistics
        model_stats = rec.get_model_statistics()
        
        # Get performance metrics
        perf_stats = performance_monitor.get_performance_stats()
        
        # Get cache statistics
        cache_stats = {
            "cache_size": len(recommendation_cache.cache),
            "max_cache_size": recommendation_cache.max_size,
            "cache_hit_rate": getattr(recommendation_cache, 'hit_rate', 0.0),
            "cache_ttl": recommendation_cache.ttl
        }
        
        # Get collaborative model stats if available
        collab_stats = {}
        try:
            if rec.collaborative_recommender.loaded:
                collab_stats = rec.collaborative_recommender.get_model_statistics()
        except:
            pass
        
        return {
            "system_info": {
                "version": "2.1.0",
                "uptime": time.time() - performance_monitor.start_time if hasattr(performance_monitor, 'start_time') else 0,
                "features_enabled": getattr(config, 'FEATURES', {})
            },
            "model_statistics": model_stats,
            "collaborative_statistics": collab_stats,
            "performance_metrics": perf_stats,
            "cache_statistics": cache_stats,
            "thresholds": {
                "collaborative_threshold_low": config.COLLABORATIVE_THRESHOLD_LOW,
                "collaborative_threshold_medium": getattr(config, 'COLLABORATIVE_THRESHOLD_MEDIUM', 5),
                "collaborative_threshold_high": getattr(config, 'COLLABORATIVE_THRESHOLD_HIGH', 8),
                "collaborative_threshold_expert": getattr(config, 'COLLABORATIVE_THRESHOLD_EXPERT', 15)
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error retrieving system statistics")

@app.post("/api/analytics/clear-cache")
async def clear_system_cache():
    """Clear all caches"""
    try:
        recommendation_cache.clear()
        
        # Clear content recommender cache if available
        if recommender and hasattr(recommender.content_recommender, 'clear_cache'):
            recommender.content_recommender.clear_cache()
        
        return {
            "message": "All caches cleared successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error clearing cache")

@app.get("/api/analytics/user-distribution")
async def get_user_distribution():
    """Get distribution of user types"""
    try:
        if not user_ids_cache:
            return {"error": "No user data available"}
        
        # Sample users for analysis (if too many)
        sample_users = user_ids_cache[:500] if len(user_ids_cache) > 500 else user_ids_cache
        
        distribution = {
            "new_user": 0,
            "light_user": 0,
            "moderate_user": 0,
            "active_user": 0,
            "power_user": 0
        }
        
        for user_id in sample_users:
            user_type = determine_user_type(user_id)
            if user_type in distribution:
                distribution[user_type] += 1
        
        return {
            "distribution": distribution,
            "total_analyzed": len(sample_users),
            "total_available": len(user_ids_cache)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error analyzing user distribution")

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.DEBUG else "An unexpected error occurred",
            "timestamp": time.time()
        }
    )

# --- Main Application Entry Point ---
if __name__ == "__main__":
    print("Starting FilmFusion Enhanced Recommendation API...")
    print(f"- Enhanced collaborative filtering with threshold: {config.COLLABORATIVE_THRESHOLD_LOW}")
    print(f"- User-aware content filtering enabled")
    print(f"- Cache size: {config.CACHE_SIZE}, TTL: {config.CACHE_TTL}s")
    
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
        log_level="info"
    )