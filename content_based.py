import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import traceback
import ast

try:
    import config
except ImportError:
    class Config:
        BASE_DIR = Path(__file__).parent
        DATA_DIR = BASE_DIR / "data"
        PROCESSED_DATA_DIR = DATA_DIR / "processed"
        MODELS_DIR = DATA_DIR / "models"
        FILES = {
            'movies_clean': PROCESSED_DATA_DIR / "movies_clean.csv",
            'tfidf_matrix': MODELS_DIR / "tfidf_matrix.pkl",
            'cosine_sim_matrix': MODELS_DIR / "cosine_sim_matrix.pkl",
            'movie_indices': MODELS_DIR / "movie_indices.pkl"
        }
    config = Config()

class ContentBasedRecommender:
    def __init__(self):
        self.movies = None
        self.tfidf_matrix = None
        self.cosine_sim_matrix = None
        self.movie_indices = None
        self.loaded = False
        self._genre_cache = {}
        
    def load_models(self):
        """Load pre-trained content-based models with improved error handling"""
        try:
            print("Loading content-based recommender...")
            
            # Load movies data with fallback paths
            self.movies = self._load_movies_data()
            if self.movies is None:
                self.loaded = False
                return
            
            # Try to load pre-trained models
            models_loaded = self._load_pretrained_models()
            
            # Create models if missing or corrupted
            if models_loaded < 3:
                print(f"Only {models_loaded}/3 models loaded. Creating from data...")
                self._create_content_models()
            
            self.loaded = True
            print("Content-based recommender loaded successfully!")
            
        except Exception as e:
            print(f"Error loading content-based models: {e}")
            traceback.print_exc()
            self.loaded = False
            
    def _load_movies_data(self) -> Optional[pd.DataFrame]:
        """Load movies data with multiple fallback paths"""
        movies_path = config.FILES['movies_clean']
        
        # Try primary path
        if os.path.exists(movies_path):
            return self._load_and_validate_movies(movies_path)
        
        # Try alternative paths
        alt_paths = [
            "data/processed/movies_clean.csv",
            "movies_clean.csv",
            Path(__file__).parent / "data" / "processed" / "movies_clean.csv"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                return self._load_and_validate_movies(alt_path)
                
        print("Error: Could not find movies_clean.csv in any expected location")
        return None
    
    def _load_and_validate_movies(self, path) -> Optional[pd.DataFrame]:
        """Load and validate movies data"""
        try:
            movies = pd.read_csv(path)
            print(f"Loaded {len(movies)} movies from {path}")
            
            # Validate required columns
            required_columns = ['id', 'title']
            missing_columns = [col for col in required_columns if col not in movies.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
            
            # Clean and validate movie IDs
            movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
            movies = movies.dropna(subset=['id'])
            movies['id'] = movies['id'].astype(int)
            
            return movies
            
        except Exception as e:
            print(f"Error loading movies from {path}: {e}")
            return None
    
    def _load_pretrained_models(self) -> int:
        """Load pre-trained models and return count of successful loads"""
        models_loaded = 0
        
        # Load TF-IDF matrix
        self.tfidf_matrix = self._safe_load_pickle(
            config.FILES['tfidf_matrix'], "TF-IDF matrix"
        )
        if self.tfidf_matrix is not None:
            models_loaded += 1
            
        # Load cosine similarity matrix
        self.cosine_sim_matrix = self._safe_load_pickle(
            config.FILES['cosine_sim_matrix'], "cosine similarity matrix"
        )
        if self.cosine_sim_matrix is not None:
            models_loaded += 1
            
        # Load movie indices
        self.movie_indices = self._safe_load_pickle(
            config.FILES['movie_indices'], "movie indices"
        )
        if self.movie_indices is not None:
            models_loaded += 1
        
        return models_loaded
        
    def _safe_load_pickle(self, file_path, description="file"):
        """Safely load pickle file with error handling"""
        try:
            if not os.path.exists(file_path):
                print(f"Warning: {description} not found at {file_path}")
                return None
                
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                print(f"Successfully loaded {description}")
                return data
        except Exception as e:
            print(f"Error loading {description} from {file_path}: {e}")
            return None
        
    def _create_content_models(self):
        """Create content-based models from movies data"""
        try:
            if self.movies is None or len(self.movies) == 0:
                print("No movies data available to create models")
                return
                
            print("Creating content-based models from movies data...")
            
            # Prepare content features
            content_features = self._extract_content_features()
            
            # Create TF-IDF matrix
            print("Creating TF-IDF matrix...")
            tfidf = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            self.tfidf_matrix = tfidf.fit_transform(content_features)
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            
            # Create cosine similarity matrix
            print("Computing cosine similarity matrix...")
            self.cosine_sim_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            print(f"Cosine similarity matrix shape: {self.cosine_sim_matrix.shape}")
            
            # Create movie indices mapping
            self.movie_indices = {}
            for idx, movie_id in enumerate(self.movies['id']):
                self.movie_indices[int(movie_id)] = idx
            
            print(f"Created movie indices for {len(self.movie_indices)} movies")
            
            # Save models for future use
            self._save_models()
            
        except Exception as e:
            print(f"Error creating content models: {e}")
            traceback.print_exc()
    
    def _extract_content_features(self) -> List[str]:
        """Extract and combine content features for each movie"""
        content_features = []
        
        for _, movie in self.movies.iterrows():
            features = []
            
            # Add title (weighted more heavily)
            if pd.notna(movie.get('title')):
                title_text = str(movie['title']).replace('-', ' ')
                features.append(f"{title_text} {title_text}")  # Repeat for weight
            
            # Add overview/description
            overview_text = ""
            if pd.notna(movie.get('overview')):
                overview_text = str(movie['overview'])
            elif pd.notna(movie.get('description')):
                overview_text = str(movie['description'])
            
            if overview_text:
                features.append(overview_text)
            
            # Add genres (heavily weighted)
            genres_text = self._extract_genres_text(movie)
            if genres_text:
                # Repeat genres 3 times for higher weight
                features.extend([genres_text] * 3)
            
            # Add director (if available)
            if pd.notna(movie.get('director')):
                director = str(movie['director']).replace(' ', '_')  # Treat as single token
                features.append(f"{director} {director}")  # Repeat for weight
            
            # Add cast (if available)
            cast_text = self._extract_cast_text(movie)
            if cast_text:
                features.append(cast_text)
            
            # Add keywords (if available)
            keywords_text = self._extract_keywords_text(movie)
            if keywords_text:
                features.append(keywords_text)
            
            # Combine all features
            combined_features = ' '.join(features).lower()
            content_features.append(combined_features)
        
        return content_features
    
    def _extract_genres_text(self, movie) -> str:
        """Extract genres as text"""
        if pd.notna(movie.get('genre_names')):
            genres = movie['genre_names']
            if isinstance(genres, str):
                try:
                    # Handle string representation of list
                    if genres.startswith('[') and genres.endswith(']'):
                        genres = ast.literal_eval(genres)
                    else:
                        genres = genres.strip('[]').replace("'", "").replace('"', '').split(', ')
                except:
                    return ""
            
            if isinstance(genres, list):
                # Replace spaces with underscores to treat multi-word genres as single tokens
                processed_genres = [genre.replace(' ', '_') for genre in genres if isinstance(genre, str)]
                return ' '.join(processed_genres)
        
        return ""
    
    def _extract_cast_text(self, movie) -> str:
        """Extract top cast as text"""
        if pd.notna(movie.get('top_cast')):
            cast = movie['top_cast']
            if isinstance(cast, str):
                try:
                    cast = ast.literal_eval(cast)
                except:
                    return ""
            
            if isinstance(cast, list):
                # Take top 5 cast members and replace spaces with underscores
                top_cast = [actor.replace(' ', '_') for actor in cast[:5] if isinstance(actor, str)]
                return ' '.join(top_cast)
        
        return ""
    
    def _extract_keywords_text(self, movie) -> str:
        """Extract keywords as text"""
        if pd.notna(movie.get('keyword_names')):
            keywords = movie['keyword_names']
            if isinstance(keywords, str):
                try:
                    keywords = ast.literal_eval(keywords)
                except:
                    return ""
            
            if isinstance(keywords, list):
                # Replace spaces with underscores for multi-word keywords
                processed_keywords = [kw.replace(' ', '_') for kw in keywords[:10] if isinstance(kw, str)]
                return ' '.join(processed_keywords)
        
        return ""
            
    def _save_models(self):
        """Save created models to disk"""
        try:
            models_dir = Path(config.FILES['tfidf_matrix']).parent
            models_dir.mkdir(parents=True, exist_ok=True)
            
            if self.tfidf_matrix is not None:
                with open(config.FILES['tfidf_matrix'], 'wb') as f:
                    pickle.dump(self.tfidf_matrix, f)
                print("Saved TF-IDF matrix")
            
            if self.cosine_sim_matrix is not None:
                with open(config.FILES['cosine_sim_matrix'], 'wb') as f:
                    pickle.dump(self.cosine_sim_matrix, f)
                print("Saved cosine similarity matrix")
            
            if self.movie_indices is not None:
                with open(config.FILES['movie_indices'], 'wb') as f:
                    pickle.dump(self.movie_indices, f)
                print("Saved movie indices")
                
        except Exception as e:
            print(f"Warning: Could not save models: {e}")
            
    def search_movies(self, query: str, limit: int = 20) -> List[Dict]:
        """Enhanced movie search with multiple strategies"""
        if not self.loaded or self.movies is None:
            print("Content-based recommender not loaded")
            return []
            
        try:
            results = []
            query_lower = query.lower().strip()
            
            # Strategy 1: Exact title matches
            exact_matches = self.movies[
                self.movies['title'].str.lower() == query_lower
            ]
            self._add_search_results(exact_matches, results, 1.0, 'Exact title match')
            
            # Strategy 2: Title contains query
            if len(results) < limit:
                contains_matches = self.movies[
                    (~self.movies.index.isin(exact_matches.index)) &
                    (self.movies['title'].str.lower().str.contains(query_lower, case=False, na=False))
                ]
                self._add_search_results(contains_matches, results, 0.9, 'Title match')
            
            # Strategy 3: Search in overview/description
            if len(results) < limit and 'overview' in self.movies.columns:
                overview_matches = self.movies[
                    (~self.movies.index.isin([r['original_index'] for r in results])) &
                    (self.movies['overview'].str.lower().str.contains(query_lower, case=False, na=False))
                ]
                needed = limit - len(results)
                self._add_search_results(overview_matches.head(needed), results, 0.7, 'Description match')
            
            # Strategy 4: Genre search
            if len(results) < limit:
                genre_matches = self._search_by_genre(query_lower, exclude_indices=[r['original_index'] for r in results])
                needed = limit - len(results)
                self._add_search_results(genre_matches.head(needed), results, 0.6, 'Genre match')
            
            # Sort by similarity score and limit
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            final_results = results[:limit]
            
            # Remove temporary field
            for result in final_results:
                result.pop('original_index', None)
            
            print(f"Search for '{query}' returned {len(final_results)} movies")
            return final_results
            
        except Exception as e:
            print(f"Error in movie search: {e}")
            traceback.print_exc()
            return []
    
    def _add_search_results(self, matches_df: pd.DataFrame, results: List[Dict], score: float, reason: str):
        """Add search results to the results list"""
        for _, movie in matches_df.iterrows():
            movie_dict = movie.to_dict()
            movie_dict['similarity_score'] = score
            movie_dict['recommendation_reason'] = reason
            movie_dict['original_index'] = movie.name  # Store original index
            results.append(movie_dict)
    
    def _search_by_genre(self, query: str, exclude_indices: List = None) -> pd.DataFrame:
        """Search movies by genre"""
        if exclude_indices is None:
            exclude_indices = []
        
        try:
            # Check if query matches any genre
            mask = pd.Series([False] * len(self.movies), index=self.movies.index)
            
            if 'genre_names' in self.movies.columns:
                for idx, row in self.movies.iterrows():
                    if idx in exclude_indices:
                        continue
                        
                    genres = row['genre_names']
                    if pd.notna(genres):
                        if isinstance(genres, str):
                            try:
                                genres = ast.literal_eval(genres)
                            except:
                                genres = []
                        
                        if isinstance(genres, list):
                            genre_text = ' '.join(genres).lower()
                            if query in genre_text:
                                mask.iloc[idx] = True
            
            return self.movies[mask]
            
        except Exception as e:
            print(f"Error in genre search: {e}")
            return pd.DataFrame()
            
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Get details for a specific movie"""
        if not self.loaded or self.movies is None:
            print("Content-based recommender not loaded")
            return None
            
        try:
            movie_id = int(movie_id)
            movie_matches = self.movies[self.movies['id'] == movie_id]
            
            if movie_matches.empty:
                print(f"Movie with ID {movie_id} not found")
                return None
                
            movie_dict = movie_matches.iloc[0].to_dict()
            movie_dict['similarity_score'] = 1.0
            movie_dict['recommendation_reason'] = 'Movie details'
            
            print(f"Found movie details for ID {movie_id}: {movie_dict.get('title', 'Unknown')}")
            return movie_dict
            
        except Exception as e:
            print(f"Error getting movie details for ID {movie_id}: {e}")
            return None
            
    def get_movie_recommendations(self, movie_id: int, n_recommendations: int = 20) -> List[Dict]:
        """Get content-based recommendations for a movie"""
        if not self.loaded or self.cosine_sim_matrix is None:
            print("Content-based recommender models not loaded")
            return self.get_top_rated_movies(n_recommendations)
            
        try:
            movie_id = int(movie_id)
            
            if movie_id not in self.movie_indices:
                print(f"Movie ID {movie_id} not in movie indices")
                return self.get_top_rated_movies(n_recommendations)
                
            movie_idx = self.movie_indices[movie_id]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations (excluding the movie itself)
            recommendations = []
            for idx, sim_score in sim_scores[1:n_recommendations+1]:
                if idx < len(self.movies):
                    movie = self.movies.iloc[idx].to_dict()
                    movie['similarity_score'] = float(sim_score)
                    movie['recommendation_reason'] = 'Content similarity'
                    recommendations.append(movie)
                    
            print(f"Generated {len(recommendations)} content-based recommendations")
            return recommendations
            
        except Exception as e:
            print(f"Error getting movie recommendations for ID {movie_id}: {e}")
            return self.get_top_rated_movies(n_recommendations)
            
    def get_top_rated_movies(self, n_movies: int = 20) -> List[Dict]:
        """Get top rated movies with improved sorting"""
        if not self.loaded or self.movies is None:
            print("No movies data available")
            return []
            
        try:
            # Create a composite score for better ranking
            movies_copy = self.movies.copy()
            
            # Try multiple rating columns
            rating_col = None
            for col in ['vote_average', 'rating', 'imdb_rating']:
                if col in movies_copy.columns and movies_copy[col].notna().any():
                    rating_col = col
                    break
            
            if rating_col:
                # Apply weighted rating (consider both rating and vote count)
                if 'vote_count' in movies_copy.columns:
                    # Use weighted rating formula
                    min_votes = movies_copy['vote_count'].quantile(0.7)  # 70th percentile
                    movies_copy['weighted_rating'] = (
                        (movies_copy['vote_count'] / (movies_copy['vote_count'] + min_votes)) * 
                        movies_copy[rating_col] +
                        (min_votes / (movies_copy['vote_count'] + min_votes)) * 
                        movies_copy[rating_col].mean()
                    )
                    sort_column = 'weighted_rating'
                else:
                    sort_column = rating_col
                    
                top_movies = movies_copy.nlargest(n_movies, sort_column)
            else:
                # If no rating column, sort by year (newer first) or just take first n
                if 'year' in movies_copy.columns:
                    top_movies = movies_copy.sort_values('year', ascending=False).head(n_movies)
                else:
                    top_movies = movies_copy.head(n_movies)
            
            recommendations = []
            for _, movie in top_movies.iterrows():
                movie_dict = movie.to_dict()
                score = movie_dict.get('weighted_rating', movie_dict.get(rating_col if rating_col else 'vote_average', 7.0))
                movie_dict['similarity_score'] = min(1.0, score / 10.0) if score else 0.7
                movie_dict['recommendation_reason'] = 'Top rated'
                recommendations.append(movie_dict)
                
            print(f"Generated {len(recommendations)} top-rated movies")
            return recommendations
            
        except Exception as e:
            print(f"Error getting top rated movies: {e}")
            return []
            
    def get_movies_by_genre(self, genre: str, n_movies: int = 20) -> List[Dict]:
        """Get movies by genre with caching"""
        if not self.loaded or self.movies is None:
            return []
        
        # Check cache first
        cache_key = f"{genre.lower()}_{n_movies}"
        if cache_key in self._genre_cache:
            return self._genre_cache[cache_key]
            
        try:
            # Filter movies by genre
            genre_movies = self._filter_by_genre(genre)
            
            if genre_movies.empty:
                print(f"No movies found for genre: {genre}")
                result = self.get_top_rated_movies(n_movies)
                self._genre_cache[cache_key] = result
                return result
            
            # Sort by composite score
            if 'vote_average' in genre_movies.columns and 'vote_count' in genre_movies.columns:
                min_votes = genre_movies['vote_count'].quantile(0.6)
                genre_movies = genre_movies.copy()
                genre_movies['genre_score'] = (
                    (genre_movies['vote_count'] / (genre_movies['vote_count'] + min_votes)) * 
                    genre_movies['vote_average'] +
                    (min_votes / (genre_movies['vote_count'] + min_votes)) * 
                    genre_movies['vote_average'].mean()
                )
                genre_movies = genre_movies.nlargest(n_movies, 'genre_score')
            else:
                # Fallback sorting
                sort_col = 'vote_average' if 'vote_average' in genre_movies.columns else None
                if sort_col:
                    genre_movies = genre_movies.nlargest(n_movies, sort_col)
                else:
                    genre_movies = genre_movies.head(n_movies)
            
            recommendations = []
            for _, movie in genre_movies.iterrows():
                movie_dict = movie.to_dict()
                rating = movie_dict.get('vote_average', movie_dict.get('rating', 7.0))
                movie_dict['similarity_score'] = min(1.0, rating / 10.0) if rating else 0.7
                movie_dict['recommendation_reason'] = f'{genre} genre'
                recommendations.append(movie_dict)
                
            print(f"Generated {len(recommendations)} movies for genre: {genre}")
            
            # Cache the result
            self._genre_cache[cache_key] = recommendations
            return recommendations
            
        except Exception as e:
            print(f"Error getting movies by genre {genre}: {e}")
            return []
    
    def _filter_by_genre(self, genre: str) -> pd.DataFrame:
        """Filter movies by genre"""
        genre_lower = genre.lower()
        
        if 'genre_names' in self.movies.columns:
            mask = pd.Series([False] * len(self.movies), index=self.movies.index)
            
            for idx, row in self.movies.iterrows():
                genres = row['genre_names']
                if pd.notna(genres):
                    if isinstance(genres, str):
                        try:
                            genres = ast.literal_eval(genres)
                        except:
                            continue
                    
                    if isinstance(genres, list):
                        genre_text = ' '.join(genres).lower()
                        if genre_lower in genre_text:
                            mask.iloc[idx] = True
            
            return self.movies[mask]
        elif 'genres' in self.movies.columns:
            # Fallback to 'genres' column
            return self.movies[
                self.movies['genres'].str.contains(genre, case=False, na=False)
            ]
        else:
            print("No genre column found")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear the genre cache"""
        self._genre_cache.clear()
        print("Content recommender cache cleared")