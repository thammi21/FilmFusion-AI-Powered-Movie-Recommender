import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Data files
FILES = {
    # Raw data files
    'movies_metadata': RAW_DATA_DIR / "movies_metadata.csv",
    'ratings': RAW_DATA_DIR / "ratings.csv",
    'credits': RAW_DATA_DIR / "credits.csv",
    'keywords': RAW_DATA_DIR / "keywords.csv",
    
    # Processed files
    'movies_clean': PROCESSED_DATA_DIR / "movies_clean.csv",
    'ratings_clean': PROCESSED_DATA_DIR / "ratings_clean_filtered.csv",
    'content_features': PROCESSED_DATA_DIR / "content_features.csv",
    
    # Model files
    'tfidf_matrix': MODELS_DIR / "tfidf_matrix.pkl",
    'cosine_sim_matrix': MODELS_DIR / "cosine_sim_matrix.pkl",
    'movie_indices': MODELS_DIR / "movie_indices_fixed.pkl",
    'collaborative_model': MODELS_DIR / "collaborative_model_fixed.pkl",
    'user_indices': MODELS_DIR / "user_indices_fixed.pkl"
}

# Data preprocessing parameters
MIN_RATINGS_PER_MOVIE = 30  # Reduced from 50 for more diversity
MIN_RATINGS_PER_USER = 10   # Reduced from 20 to include more users
MIN_YEAR = 1980
MAX_FEATURES = 8000         # Increased from 10000 for better content modeling
CONTENT_SIMILARITY_THRESHOLD = 0.05  # Lowered threshold

# Hybrid model parameters - Updated for better balance
CONTENT_WEIGHT = 0.4        # Reduced default content weight
COLLABORATIVE_WEIGHT = 0.6  # Increased default collaborative weight

# Collaborative filtering parameters - Optimized
MIN_USER_RATINGS = 3        # Significantly reduced from 20
MIN_MOVIE_RATINGS = 5       # Reduced from 10
N_FACTORS = 50              # Kept optimal
REGULARIZATION = 0.1
LEARNING_RATE = 0.01
N_EPOCHS = 20

# Dynamic weight thresholds - New parameters for improved hybrid logic
COLLABORATIVE_THRESHOLD_LOW = 3      # Start using collaborative
COLLABORATIVE_THRESHOLD_MEDIUM = 5   # Moderate collaborative preference
COLLABORATIVE_THRESHOLD_HIGH = 8     # Strong collaborative preference
COLLABORATIVE_THRESHOLD_EXPERT = 15  # Heavy collaborative preference

# Weight adjustments for different user types
WEIGHTS = {
    'new_user': {'content': 0.7, 'collaborative': 0.3},
    'light_user': {'content': 0.45, 'collaborative': 0.55},      # 3-5 ratings
    'moderate_user': {'content': 0.4, 'collaborative': 0.6},     # 5-8 ratings
    'active_user': {'content': 0.3, 'collaborative': 0.7},      # 8-15 ratings
    'power_user': {'content': 0.2, 'collaborative': 0.8}        # 15+ ratings
}

# Content-based improvements
CONTENT_FEATURE_WEIGHTS = {
    'title': 2,           # Title repetitions for weight
    'genres': 3,          # Genre repetitions for weight
    'director': 2,        # Director repetitions
    'overview': 1,        # Overview weight
    'keywords': 1,        # Keywords weight
    'cast': 1            # Cast weight
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG = True

# Performance settings
CACHE_SIZE = 2000          # Increased cache size
CACHE_TTL = 7200          # 2 hours (increased from 1 hour)
MAX_RECOMMENDATIONS = 50   # Maximum recommendations to generate
DEFAULT_RECOMMENDATIONS = 20

# Cold start settings
COLD_START_MOVIES_COUNT = 15
ONBOARDING_THRESHOLD = 2     # Reduced from 3

# Search and filtering
MAX_SEARCH_RESULTS = 30
SEARCH_SIMILARITY_THRESHOLD = 0.1
GENRE_CACHE_SIZE = 100
MIN_GENRE_MOVIES = 2         # Minimum movies in genre for preference

# Quality thresholds
MIN_MOVIE_RATING = 4.0       # Minimum rating for quality boost
HIGH_MOVIE_RATING = 7.5      # High rating threshold
EXCELLENT_MOVIE_RATING = 8.5 # Excellent rating threshold

# Popularity thresholds  
MIN_VOTE_COUNT = 100         # Minimum votes for popularity boost
POPULAR_VOTE_COUNT = 1000    # Popular movie threshold

# Recommendation diversity
MAX_SAME_GENRE = 3           # Maximum movies from same genre
DIVERSITY_BOOST = 1.1        # Score boost for diverse recommendations
CONSENSUS_BOOST = 1.1        # Score boost when both systems agree

# TMDB API (optional)
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_PERFORMANCE = True
LOG_RECOMMENDATIONS = False  # Set to True for debugging

# Feature flags for A/B testing
FEATURES = {
    'user_aware_content_filtering': True,
    'dynamic_weight_calculation': True,
    'quality_boosting': True,
    'popularity_boosting': True,
    'genre_diversification': True,
    'collaborative_early_start': True,
    'enhanced_similarity_scoring': True
}

# Create directories if they don't exist
def ensure_directories():
    """Ensure all required directories exist"""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    print("Directory structure verified")

# Validation functions
def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Check weight consistency
    if abs(CONTENT_WEIGHT + COLLABORATIVE_WEIGHT - 1.0) > 0.01:
        errors.append("Content and collaborative weights must sum to 1.0")
    
    # Check thresholds are increasing
    thresholds = [
        COLLABORATIVE_THRESHOLD_LOW,
        COLLABORATIVE_THRESHOLD_MEDIUM, 
        COLLABORATIVE_THRESHOLD_HIGH,
        COLLABORATIVE_THRESHOLD_EXPERT
    ]
    
    if thresholds != sorted(thresholds):
        errors.append("Collaborative thresholds must be in ascending order")
    
    # Check minimum values
    if MIN_USER_RATINGS < 1:
        errors.append("MIN_USER_RATINGS must be at least 1")
        
    if MIN_MOVIE_RATINGS < 1:
        errors.append("MIN_MOVIE_RATINGS must be at least 1")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("Configuration validated successfully")
    return True

# Initialize on import
if __name__ == "__main__":
    ensure_directories()
    validate_config()
else:
    ensure_directories()