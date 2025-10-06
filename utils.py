import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import time
from functools import lru_cache
import ast
import math

# Import config with fallback
try:
    import config
except ImportError:
    # Fallback config values
    class Config:
        TMDB_API_KEY = "TMDB_API_KEY"
        TMDB_BASE_URL = "https://api.themoviedb.org/3"
        TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
    config = Config()

class TMDBApi:
    """Interface for The Movie Database API to fetch movie posters and additional info"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or getattr(config, 'TMDB_API_KEY', '')
        self.base_url = getattr(config, 'TMDB_BASE_URL', 'https://api.themoviedb.org/3')
        self.image_base_url = getattr(config, 'TMDB_IMAGE_BASE_URL', 'https://image.tmdb.org/t/p/w500')
        self.session = requests.Session()
        # Set headers for TMDB API
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json;charset=utf-8'
            })
        
    @lru_cache(maxsize=2000)
    def get_movie_poster(self, movie_title: str, year: Optional[int] = None) -> Optional[str]:
        """Get movie poster URL from TMDB"""
        if not self.api_key:
            return None
            
        try:
            # Search for movie
            search_url = f"{self.base_url}/search/movie"
            params = {
                'query': movie_title
            }
            
            if year:
                params['year'] = year
                
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    poster_path = data['results'][0].get('poster_path')
                    if poster_path:
                        return f"{self.image_base_url}{poster_path}"
                        
            return None
            
        except Exception as e:
            return None
            
    @lru_cache(maxsize=1000)
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed movie information from TMDB"""
        if not self.api_key:
            return None
            
        try:
            url = f"{self.base_url}/movie/{tmdb_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
                
            return None
            
        except Exception as e:
            return None

class RecommendationCache:
    """Simple in-memory cache for recommendations"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
        
    def get(self, key: str) -> Optional[any]:
        """Get item from cache"""
        current_time = time.time()
        
        if key in self.cache:
            # Check if expired
            if current_time - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
                
            return self.cache[key]
            
        return None
        
    def set(self, key: str, value: any):
        """Set item in cache"""
        current_time = time.time()
        
        # Clean expired items
        self._clean_expired()
        
        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
            
        self.cache[key] = value
        self.timestamps[key] = current_time
        
    def _clean_expired(self):
        """Remove expired items from cache"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
            
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.timestamps.clear()

class MoviePosterManager:
    """Manage movie posters with caching and fallbacks"""
    
    def __init__(self):
        self.tmdb_api = TMDBApi()
        self.poster_cache = RecommendationCache(max_size=3000, ttl=86400)  # 24 hour cache
        
    def get_poster_url(self, movie_title: str, year: Optional[int] = None, movie_id: Optional[int] = None) -> str:
        """Get poster URL with fallback to default poster"""
        cache_key = f"poster_{movie_title}_{year}_{movie_id}"
        
        # Check cache first
        cached_url = self.poster_cache.get(cache_key)
        if cached_url:
            return cached_url
            
        # Try to get from TMDB
        poster_url = self.tmdb_api.get_movie_poster(movie_title, year)
        
        if poster_url:
            self.poster_cache.set(cache_key, poster_url)
            return poster_url
            
        # Fallback to default poster
        default_poster = "https://via.placeholder.com/500x750/333/fff?text=No+Poster"
        self.poster_cache.set(cache_key, default_poster)
        return default_poster

# Initialize global poster manager
poster_manager = MoviePosterManager()

class DataValidator:
    """Validate data integrity and model consistency"""
    
    @staticmethod
    def validate_movie_data(movies_df: pd.DataFrame) -> Dict:
        """Validate movies dataset"""
        validation_results = {
            'total_movies': len(movies_df),
            'missing_titles': movies_df['title'].isna().sum(),
            'missing_overviews': movies_df['overview'].isna().sum(),
            'missing_years': movies_df['year'].isna().sum(),
            'invalid_ratings': len(movies_df[
                (movies_df['vote_average'] < 0) | (movies_df['vote_average'] > 10)
            ]),
            'duplicate_ids': movies_df['id'].duplicated().sum()
        }
        
        return validation_results
        
    @staticmethod
    def validate_ratings_data(ratings_df: pd.DataFrame) -> Dict:
        """Validate ratings dataset"""
        validation_results = {
            'total_ratings': len(ratings_df),
            'unique_users': ratings_df['userId'].nunique(),
            'unique_movies': ratings_df['movieId'].nunique(),
            'invalid_ratings': len(ratings_df[
                (ratings_df['rating'] < 0.5) | (ratings_df['rating'] > 5.0)
            ]),
            'missing_values': ratings_df.isna().sum().sum(),
            'rating_distribution': ratings_df['rating'].value_counts().to_dict()
        }
        
        return validation_results

class PerformanceMonitor:
    """Monitor recommendation system performance"""
    
    def __init__(self):
        self.metrics = {
            'recommendation_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0
        }
        
    def log_recommendation_time(self, duration: float):
        """Log time taken for recommendation"""
        self.metrics['recommendation_times'].append(duration)
        
    def log_cache_hit(self):
        """Log cache hit"""
        self.metrics['cache_hits'] += 1
        
    def log_cache_miss(self):
        """Log cache miss"""
        self.metrics['cache_misses'] += 1
        
    def log_api_call(self):
        """Log API call"""
        self.metrics['api_calls'] += 1
        
    def log_error(self):
        """Log error"""
        self.metrics['errors'] += 1
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        rec_times = self.metrics['recommendation_times']
        
        stats = {
            'average_recommendation_time': np.mean(rec_times) if rec_times else 0,
            'median_recommendation_time': np.median(rec_times) if rec_times else 0,
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0,
            'total_api_calls': self.metrics['api_calls'],
            'total_errors': self.metrics['errors'],
            'total_recommendations': len(rec_times)
        }
        
        return stats

# User rating storage for in-memory persistence
user_ratings_storage = {}

def store_user_rating(user_id: int, movie_id: int, rating: float):
    """Store user rating in memory"""
    if user_id not in user_ratings_storage:
        user_ratings_storage[user_id] = {}
    
    user_ratings_storage[user_id][movie_id] = {
        'rating': rating,
        'timestamp': time.time()
    }

def get_user_ratings(user_id: int) -> Dict:
    """Get all ratings for a user"""
    return user_ratings_storage.get(user_id, {})

def safe_parse_list(val):
    """Safely parse a value that should be a list"""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        if not val.strip():
            return []
        try:
            # Handle string representations of lists
            if val.startswith('[') and val.endswith(']'):
                parsed = ast.literal_eval(val)
                return parsed if isinstance(parsed, list) else []
            else:
                # Split by comma for simple comma-separated values
                return [item.strip() for item in val.split(',') if item.strip()]
        except (ValueError, SyntaxError):
            return []
    return []

def safe_float_conversion(value, default=0.0) -> float:
    """Safely convert value to float"""
    if value is None or value == "":
        return default
    try:
        val = float(value)
        return default if math.isnan(val) or math.isinf(val) else val
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value, default=0) -> int:
    """Safely convert value to int"""
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def get_default_poster_path() -> str:
    """Get path to default poster image"""
    return "https://via.placeholder.com/500x750/333/fff?text=No+Poster"

def normalize_movie_data(movie_data: Union[Dict, pd.Series]) -> Dict:
    """Normalize movie data from various sources into a consistent format"""
    if isinstance(movie_data, pd.Series):
        movie_data = movie_data.to_dict()
    
    # Handle different possible field names and normalize them
    normalized = {
        'id': safe_int_conversion(
            movie_data.get('id') or 
            movie_data.get('movieId') or 
            movie_data.get('movie_id', 0)
        ),
        'title': str(movie_data.get('title', 'Unknown Title')),
        'year': safe_int_conversion(
            movie_data.get('year') or 
            movie_data.get('release_year') or 
            movie_data.get('release_date', '')[:4] if movie_data.get('release_date') else None
        ) if movie_data.get('year') or movie_data.get('release_year') or movie_data.get('release_date') else None,
        'rating': safe_float_conversion(
            movie_data.get('vote_average') or 
            movie_data.get('rating') or 
            movie_data.get('imdb_rating', 0.0)
        ),
        'genre_names': safe_parse_list(
            movie_data.get('genre_names') or 
            movie_data.get('genres') or 
            movie_data.get('genre_list', [])
        ),
        'overview': str(movie_data.get('overview') or movie_data.get('description', '')),
        'runtime': safe_int_conversion(movie_data.get('runtime')) if movie_data.get('runtime') else None,
        'director': str(movie_data.get('director', '')),
        'similarity_score': safe_float_conversion(
            movie_data.get('similarity_score') or 
            movie_data.get('hybrid_score') or 
            movie_data.get('score', 0.0)
        ),
        'recommendation_reason': str(
            movie_data.get('recommendation_reason') or 
            movie_data.get('reason', 'Recommended')
        ),
        'explanation': str(movie_data.get('explanation', '')),
        'poster_url': movie_data.get('poster_url')
    }
    
    return normalized

def batch_format_recommendations(recommendations):
    """Format raw recommendation data into standardized movie response objects with posters"""
    if not recommendations:
        return []
    
    formatted = []
    
    for rec in recommendations:
        try:
            # Normalize the movie data first
            normalized_rec = normalize_movie_data(rec)
            
            # Add poster URL if not present
            if not normalized_rec.get('poster_url'):
                poster_url = poster_manager.get_poster_url(
                    normalized_rec['title'],
                    normalized_rec['year'],
                    normalized_rec['id']
                )
                normalized_rec['poster_url'] = poster_url
            
            formatted.append(normalized_rec)
            
        except Exception as e:
            # Create a minimal valid movie object to prevent complete failure
            try:
                minimal_rec = {
                    'id': safe_int_conversion(rec.get('id', 0)) if isinstance(rec, dict) else 0,
                    'title': str(rec.get('title', 'Unknown Title')) if isinstance(rec, dict) else 'Unknown Title',
                    'year': None,
                    'rating': 0.0,
                    'genre_names': [],
                    'overview': '',
                    'runtime': None,
                    'director': '',
                    'similarity_score': 0.0,
                    'recommendation_reason': 'Recommended',
                    'explanation': '',
                    'poster_url': get_default_poster_path()
                }
                formatted.append(minimal_rec)
            except Exception:
                continue
    
    return formatted