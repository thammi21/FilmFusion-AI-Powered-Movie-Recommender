import pandas as pd
import numpy as np
import json
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from pathlib import Path
import config

class MovieDataPreprocessor:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.credits = None
        self.keywords = None
        
    def load_raw_data(self):
        """Load raw data from CSV files"""
        print("Loading raw data...")
        
        # Load movies metadata
        self.movies = pd.read_csv(config.FILES['movies_metadata'], low_memory=False)
        print(f"Movies loaded: {len(self.movies)} records")
        
        # Load ratings
        self.ratings = pd.read_csv(config.FILES['ratings'])
        print(f"Ratings loaded: {len(self.ratings)} records")
        
        # Load credits if available
        try:
            self.credits = pd.read_csv(config.FILES['credits'])
            print(f"Credits loaded: {len(self.credits)} records")
        except:
            print("Credits file not found, skipping...")
            self.credits = None
        
        # Load keywords if available
        try:
            self.keywords = pd.read_csv(config.FILES['keywords'])
            print(f"Keywords loaded: {len(self.keywords)} records")
        except:
            print("Keywords file not found, skipping...")
            self.keywords = None
        
    def clean_movies_data(self):
        """Clean and preprocess movies metadata"""
        print("Cleaning movies data...")
        
        # Remove rows with invalid IDs
        self.movies = self.movies[pd.to_numeric(self.movies['id'], errors='coerce').notna()]
        self.movies['id'] = self.movies['id'].astype(int)
        
        # Clean basic columns
        self.movies['title'] = self.movies['title'].fillna('')
        self.movies['overview'] = self.movies['overview'].fillna('')
        self.movies['release_date'] = pd.to_datetime(self.movies['release_date'], errors='coerce')
        self.movies['runtime'] = pd.to_numeric(self.movies['runtime'], errors='coerce')
        self.movies['vote_average'] = pd.to_numeric(self.movies['vote_average'], errors='coerce')
        self.movies['vote_count'] = pd.to_numeric(self.movies['vote_count'], errors='coerce')
        
        # Extract year from release_date
        self.movies['year'] = self.movies['release_date'].dt.year
        
        # Parse JSON columns
        self.movies['genres'] = self.movies['genres'].apply(self._parse_json_column)
        self.movies['production_companies'] = self.movies['production_companies'].apply(self._parse_json_column)
        self.movies['production_countries'] = self.movies['production_countries'].apply(self._parse_json_column)
        self.movies['spoken_languages'] = self.movies['spoken_languages'].apply(self._parse_json_column)
        
        # Extract genre names
        self.movies['genre_names'] = self.movies['genres'].apply(
            lambda x: [genre['name'] for genre in x] if isinstance(x, list) else []
        )
        
        # Filter movies with minimum vote count (relaxed to include more movies)
        min_votes = getattr(config, 'MIN_RATINGS_PER_MOVIE', 30)
        self.movies = self.movies[self.movies['vote_count'] >= min_votes]
        
        print(f"Movies after cleaning: {len(self.movies)}")
        
    def clean_ratings_data(self):
        """Clean and preprocess ratings data"""
        print("Cleaning ratings data...")
        
        # Convert types
        self.ratings['userId'] = self.ratings['userId'].astype(int)
        self.ratings['movieId'] = self.ratings['movieId'].astype(int)
        self.ratings['rating'] = self.ratings['rating'].astype(float)
        
        # Filter users with minimum ratings
        min_user_ratings = getattr(config, 'MIN_RATINGS_PER_USER', 10)
        user_counts = self.ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        self.ratings = self.ratings[self.ratings['userId'].isin(active_users)]
        
        # CRITICAL: Filter movies that exist in movies dataset
        self.ratings = self.ratings[self.ratings['movieId'].isin(self.movies['id'])]
        
        # CRITICAL: Filter movies dataset to only include movies with ratings
        # This ensures alignment between content-based and collaborative models
        rated_movie_ids = self.ratings['movieId'].unique()
        self.movies = self.movies[self.movies['id'].isin(rated_movie_ids)]
        print(f"Movies filtered to those with ratings: {len(self.movies)}")
        
        print(f"Ratings after cleaning: {len(self.ratings)}")
        print(f"Unique users: {self.ratings['userId'].nunique()}")
        print(f"Unique movies: {self.ratings['movieId'].nunique()}")
        
    def merge_additional_data(self):
        """Merge credits and keywords data"""
        print("Merging additional data...")
        
        # Clean and merge credits if available
        if self.credits is not None:
            try:
                self.credits['id'] = pd.to_numeric(self.credits['id'], errors='coerce')
                self.credits = self.credits.dropna(subset=['id'])
                self.credits['id'] = self.credits['id'].astype(int)
                
                # Parse cast and crew
                self.credits['cast'] = self.credits['cast'].apply(self._parse_json_column)
                self.credits['crew'] = self.credits['crew'].apply(self._parse_json_column)
                
                # Extract director and top cast
                self.credits['director'] = self.credits['crew'].apply(self._extract_director)
                self.credits['top_cast'] = self.credits['cast'].apply(
                    lambda x: [person['name'] for person in x[:5]] if isinstance(x, list) else []
                )
                
                # Merge with movies
                self.movies = self.movies.merge(
                    self.credits[['id', 'director', 'top_cast']], 
                    on='id', 
                    how='left'
                )
                print("Credits merged successfully")
            except Exception as e:
                print(f"Error merging credits: {e}")
        
        # Clean and merge keywords if available
        if self.keywords is not None:
            try:
                self.keywords['id'] = pd.to_numeric(self.keywords['id'], errors='coerce')
                self.keywords = self.keywords.dropna(subset=['id'])
                self.keywords['id'] = self.keywords['id'].astype(int)
                
                self.keywords['keywords'] = self.keywords['keywords'].apply(self._parse_json_column)
                self.keywords['keyword_names'] = self.keywords['keywords'].apply(
                    lambda x: [kw['name'] for kw in x] if isinstance(x, list) else []
                )
                
                # Merge with movies
                self.movies = self.movies.merge(
                    self.keywords[['id', 'keyword_names']], 
                    on='id', 
                    how='left'
                )
                print("Keywords merged successfully")
            except Exception as e:
                print(f"Error merging keywords: {e}")
        
        # Fill missing values
        if 'director' not in self.movies.columns:
            self.movies['director'] = ''
        if 'top_cast' not in self.movies.columns:
            self.movies['top_cast'] = [[] for _ in range(len(self.movies))]
        if 'keyword_names' not in self.movies.columns:
            self.movies['keyword_names'] = [[] for _ in range(len(self.movies))]
            
        self.movies['director'] = self.movies['director'].fillna('')
        self.movies['top_cast'] = self.movies['top_cast'].fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        self.movies['keyword_names'] = self.movies['keyword_names'].fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        
    def create_content_features(self):
        """Create content-based features for recommendations"""
        print("Creating content features...")
        
        # CRITICAL: Reset index to ensure alignment
        self.movies = self.movies.reset_index(drop=True)
        
        # Create combined text features
        self.movies['combined_features'] = self.movies.apply(self._combine_features, axis=1)
        
        # Create TF-IDF matrix
        max_features = getattr(config, 'MAX_FEATURES', 8000)
        tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = tfidf.fit_transform(self.movies['combined_features'])
        
        # Calculate cosine similarity
        print("Computing cosine similarity matrix...")
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # CRITICAL FIX: Create movie indices based on POSITION in filtered dataset
        # This ensures indices 0, 1, 2... match the movies that have ratings
        movie_indices = {int(movie_id): idx 
                        for idx, movie_id in enumerate(self.movies['id'])}
        
        print(f"Movie indices created for {len(movie_indices)} movies")
        
        # Save models using pickle for consistency
        with open(config.FILES['tfidf_matrix'], 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(config.FILES['cosine_sim_matrix'], 'wb') as f:
            pickle.dump(cosine_sim, f)
        with open(config.FILES['movie_indices'], 'wb') as f:
            pickle.dump(movie_indices, f)
        
        print(f"Content features created. Matrix shape: {tfidf_matrix.shape}")
        
        return movie_indices
        
    def create_collaborative_model(self):
        """Create collaborative filtering model"""
        print("Creating collaborative filtering model...")
        
        # Create user-movie matrix
        user_movie_pivot = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        print(f"User-movie matrix shape: {user_movie_pivot.shape}")
        
        # Convert to sparse matrix
        user_movie_matrix = csr_matrix(user_movie_pivot.values)
        
        # Create indices mappings
        user_indices = {user: idx for idx, user in enumerate(user_movie_pivot.index)}
        movie_indices_collab = {movie: idx for idx, movie in enumerate(user_movie_pivot.columns)}
        
        print(f"Users: {len(user_indices)}, Movies: {len(movie_indices_collab)}")
        
        # Train SVD model
        n_factors = getattr(config, 'N_FACTORS', 50)
        n_components = min(n_factors, user_movie_matrix.shape[1] - 1, 100)
        
        svd_model = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            algorithm='randomized'
        )
        
        svd_model.fit(user_movie_matrix)
        
        print(f"SVD trained with {n_components} components")
        print(f"Explained variance: {svd_model.explained_variance_ratio_.sum():.3f}")
        
        # Save collaborative model
        model_data = {
            'model': svd_model,
            'user_movie_matrix': user_movie_matrix,
            'movie_indices': movie_indices_collab,  # CRITICAL: Save with model
            'user_factors': svd_model.transform(user_movie_matrix),
            'movie_factors': svd_model.components_.T,
            'n_components': n_components,
            'explained_variance_ratio': svd_model.explained_variance_ratio_
        }
        
        with open(config.FILES['collaborative_model'], 'wb') as f:
            pickle.dump(model_data, f)
        
        with open(config.FILES['user_indices'], 'wb') as f:
            pickle.dump(user_indices, f)
        
        print("Collaborative model saved successfully")
        
        return movie_indices_collab
        
    def verify_alignment(self, content_indices, collab_indices):
        """Verify that content and collaborative indices are aligned"""
        print("\nVerifying index alignment...")
        
        content_movies = set(content_indices.keys())
        collab_movies = set(collab_indices.keys())
        
        common_movies = content_movies & collab_movies
        content_only = content_movies - collab_movies
        collab_only = collab_movies - content_movies
        
        print(f"Movies in content indices: {len(content_movies)}")
        print(f"Movies in collaborative indices: {len(collab_movies)}")
        print(f"Movies in both: {len(common_movies)}")
        print(f"Content only: {len(content_only)}")
        print(f"Collaborative only: {len(collab_only)}")
        
        if len(content_only) > 0 or len(collab_only) > 0:
            print("WARNING: Indices are not perfectly aligned!")
            print("This may cause evaluation issues.")
        else:
            print("SUCCESS: All indices are perfectly aligned!")
            
        return len(content_only) == 0 and len(collab_only) == 0
        
    def save_processed_data(self):
        """Save cleaned and processed data"""
        print("\nSaving processed data...")
        
        # Select relevant columns for movies
        movies_columns = [
            'id', 'title', 'overview', 'genres', 'genre_names', 'release_date', 'year',
            'runtime', 'vote_average', 'vote_count', 'director', 'top_cast', 'keyword_names',
            'combined_features'
        ]
        
        # Only save columns that exist
        existing_columns = [col for col in movies_columns if col in self.movies.columns]
        self.movies[existing_columns].to_csv(config.FILES['movies_clean'], index=False)
        
        self.ratings.to_csv(config.FILES['ratings_clean'], index=False)
        
        # Create content features summary
        content_features = self.movies[['id', 'title', 'combined_features']].copy()
        content_features.to_csv(config.FILES['content_features'], index=False)
        
        print("Data preprocessing completed!")
        
    def _parse_json_column(self, x):
        """Parse JSON string column"""
        if pd.isna(x):
            return []
        try:
            return json.loads(x.replace("'", '"'))
        except:
            try:
                return ast.literal_eval(x)
            except:
                return []
                
    def _extract_director(self, crew_list):
        """Extract director from crew list"""
        if isinstance(crew_list, list):
            for person in crew_list:
                if person.get('job') == 'Director':
                    return person.get('name', '')
        return ''
        
    def _combine_features(self, row):
        """Combine multiple features into single text with weights"""
        features = []
        
        # Get feature weights from config
        weights = getattr(config, 'CONTENT_FEATURE_WEIGHTS', {
            'title': 2, 'genres': 3, 'director': 2,
            'overview': 1, 'keywords': 1, 'cast': 1
        })
        
        # Add title with weight
        if row.get('title'):
            title = str(row['title']).replace('-', ' ')
            features.extend([title] * weights.get('title', 2))
        
        # Add overview
        if row.get('overview'):
            features.extend([str(row['overview'])] * weights.get('overview', 1))
            
        # Add genres with weight
        if row.get('genre_names'):
            genre_names = row['genre_names']
            if isinstance(genre_names, list) and genre_names:
                genre_text = ' '.join([g.replace(' ', '_') for g in genre_names])
                features.extend([genre_text] * weights.get('genres', 3))
            
        # Add director with weight
        if row.get('director'):
            director = str(row['director']).replace(' ', '_')
            features.extend([director] * weights.get('director', 2))
            
        # Add top cast
        if row.get('top_cast'):
            top_cast = row['top_cast']
            if isinstance(top_cast, list) and top_cast:
                cast_text = ' '.join([actor.replace(' ', '_') for actor in top_cast[:5]])
                features.extend([cast_text] * weights.get('cast', 1))
            
        # Add keywords
        if row.get('keyword_names'):
            keywords = row['keyword_names']
            if isinstance(keywords, list) and keywords:
                keywords_text = ' '.join([kw.replace(' ', '_') for kw in keywords[:10]])
                features.extend([keywords_text] * weights.get('keywords', 1))
            
        return ' '.join(features).lower()
        
    def run_preprocessing(self):
        """Run complete preprocessing pipeline with alignment verification"""
        self.load_raw_data()
        self.clean_movies_data()
        self.clean_ratings_data()
        self.merge_additional_data()
        
        # Create both models
        content_indices = self.create_content_features()
        collab_indices = self.create_collaborative_model()
        
        # Verify alignment
        is_aligned = self.verify_alignment(content_indices, collab_indices)
        
        self.save_processed_data()
        
        return {
            'movies_count': len(self.movies),
            'ratings_count': len(self.ratings),
            'users_count': self.ratings['userId'].nunique(),
            'unique_movies_rated': self.ratings['movieId'].nunique(),
            'indices_aligned': is_aligned
        }

if __name__ == "__main__":
    print("="*60)
    print("ENHANCED DATA PREPROCESSING WITH INDEX ALIGNMENT")
    print("="*60)
    
    preprocessor = MovieDataPreprocessor()
    stats = preprocessor.run_preprocessing()
    
    print("\n" + "="*60)
    print("PREPROCESSING STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, bool):
            print(f"{key}: {'✓ YES' if value else '✗ NO'}")
        else:
            print(f"{key}: {value:,}")
    
    if stats.get('indices_aligned', False):
        print("\n✓ SUCCESS: All indices are properly aligned!")
        
    else:
        print("\n✗ WARNING: Index alignment issues detected!")
        print("Please review the output above.")