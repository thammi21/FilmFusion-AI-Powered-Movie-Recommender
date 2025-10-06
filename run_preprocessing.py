# run_preprocessing.py

#!/usr/bin/env python3
"""
Consolidated and improved preprocessing pipeline for the FilmFusion recommender system.
This script is the single source of truth for data preparation.
"""

import pandas as pd
import numpy as np
import pickle
import time
import ast
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
try:
    import config
    print(f"Using configuration from config.py")
except ImportError:
    print("Warning: config.py not found. Using fallback configuration.")
    class Config:
        BASE_DIR = Path(__file__).parent
        DATA_DIR = BASE_DIR / "data"
        RAW_DATA_DIR = DATA_DIR / "raw"
        PROCESSED_DATA_DIR = DATA_DIR / "processed"
        MODELS_DIR = DATA_DIR / "models"
        
        FILES = {
            'movies_metadata': RAW_DATA_DIR / "movies_metadata.csv",
            'ratings': RAW_DATA_DIR / "ratings.csv",
            'credits': RAW_DATA_DIR / "credits.csv",
            'keywords': RAW_DATA_DIR / "keywords.csv",
            'movies_clean': PROCESSED_DATA_DIR / "movies_clean.csv",
            'ratings_clean': PROCESSED_DATA_DIR / "ratings_clean_filtered.csv",
            'tfidf_matrix': MODELS_DIR / "tfidf_matrix.pkl",
            'cosine_sim_matrix': MODELS_DIR / "cosine_sim_matrix.pkl",
            'movie_indices': MODELS_DIR / "movie_indices_fixed.pkl",
            'collaborative_model': MODELS_DIR / "collaborative_model_fixed.pkl",
            'user_indices': MODELS_DIR / "user_indices_fixed.pkl"
        }
        
        MIN_RATINGS_PER_USER = 10
        MIN_RATINGS_PER_MOVIE = 30
        MAX_FEATURES = 8000
        N_FACTORS = 50
        
        CONTENT_FEATURE_WEIGHTS = {
            'title': 2, 'genres': 3, 'director': 2,
            'overview': 1, 'keywords': 1, 'cast': 1
        }
    config = Config()

# --- Helper Functions ---

def safe_literal_eval(val):
    if pd.isna(val) or val == '': return []
    if isinstance(val, (list, dict)): return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []

def extract_names_from_list(val, key='name', limit=None):
    items = safe_literal_eval(val)
    if isinstance(items, list):
        names = [item.get(key) for item in items if isinstance(item, dict) and key in item]
        return names[:limit] if limit else names
    return []

def extract_director(crew_data):
    crew_list = safe_literal_eval(crew_data)
    if isinstance(crew_list, list):
        for person in crew_list:
            if isinstance(person, dict) and person.get('job') == 'Director':
                return person.get('name', '')
    return ''

# --- Pipeline Steps ---

def step_1_clean_movies_data():
    """Cleans and preprocesses the raw movies_metadata.csv file."""
    print("\n[STEP 1/5] Cleaning movies data...")
    movies_df = pd.read_csv(config.FILES['movies_metadata'], low_memory=False)
    
    movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
    movies_df.dropna(subset=['id'], inplace=True)
    movies_df['id'] = movies_df['id'].astype(int)
    movies_df.drop_duplicates(subset=['id'], inplace=True)
    
    movies_df['title'] = movies_df['title'].fillna('Unknown Title')
    movies_df['overview'] = movies_df['overview'].fillna('')
    movies_df['genre_names'] = movies_df['genres'].apply(extract_names_from_list)
    
    print(f"Cleaned movies data. Shape: {movies_df.shape}")
    return movies_df

def step_2_merge_additional_data(movies_df):
    """Merges credits and keywords data into the movies dataframe."""
    print("\n[STEP 2/5] Merging additional data (credits & keywords)...")
    
    # Merge Credits
    try:
        credits_df = pd.read_csv(config.FILES['credits'])
        credits_df['id'] = pd.to_numeric(credits_df['id'], errors='coerce').astype(int)
        credits_df['director'] = credits_df['crew'].apply(extract_director)
        credits_df['top_cast'] = credits_df['cast'].apply(lambda x: extract_names_from_list(x, limit=5))
        movies_df = movies_df.merge(credits_df[['id', 'director', 'top_cast']], on='id', how='left')
        print("Merged credits data.")
    except FileNotFoundError:
        print("Credits file not found, skipping.")
        movies_df['director'] = ''
        movies_df['top_cast'] = [[] for _ in range(len(movies_df))]

    # Merge Keywords
    try:
        keywords_df = pd.read_csv(config.FILES['keywords'])
        keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce').astype(int)
        keywords_df['keyword_names'] = keywords_df['keywords'].apply(extract_names_from_list)
        movies_df = movies_df.merge(keywords_df[['id', 'keyword_names']], on='id', how='left')
        print("Merged keywords data.")
    except FileNotFoundError:
        print("Keywords file not found, skipping.")
        movies_df['keyword_names'] = [[] for _ in range(len(movies_df))]
        
    # Fill any remaining NaNs
    movies_df['director'] = movies_df['director'].fillna('')
    for col in ['top_cast', 'keyword_names']:
        movies_df[col] = movies_df[col].apply(lambda d: d if isinstance(d, list) else [])

    return movies_df

def step_3_clean_and_filter_ratings(movies_df):
    """Cleans the ratings data and filters it based on configured thresholds."""
    print("\n[STEP 3/5] Cleaning and filtering ratings data...")
    ratings_df = pd.read_csv(config.FILES['ratings'])
    
    # Filter ratings to only include movies we have metadata for
    valid_movie_ids = set(movies_df['id'])
    ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]
    
    # Filter by user activity
    user_counts = ratings_df['userId'].value_counts()
    active_users = user_counts[user_counts >= config.MIN_RATINGS_PER_USER].index
    ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]
    
    # Filter by movie popularity
    movie_counts = ratings_df['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= config.MIN_RATINGS_PER_MOVIE].index
    ratings_df = ratings_df[ratings_df['movieId'].isin(popular_movies)]
    
    # Final cleanup
    ratings_df = ratings_df.astype({'userId': int, 'movieId': int, 'rating': float})
    print(f"Cleaned ratings data. Shape: {ratings_df.shape}")
    print(f"Unique users: {ratings_df['userId'].nunique()}, Unique movies: {ratings_df['movieId'].nunique()}")
    
    return ratings_df

def step_4_create_content_models(movies_df, ratings_df):
    """Creates and saves content-based models (TF-IDF, Cosine Similarity)."""
    print("\n[STEP 4/5] Creating content-based models...")
    
    rated_movie_ids = ratings_df['movieId'].unique()
    movies_df_filtered = movies_df[movies_df['id'].isin(rated_movie_ids)].reset_index(drop=True)
    
    def combine_features(row):
        weights = config.CONTENT_FEATURE_WEIGHTS
        title = ' '.join([row['title']] * weights.get('title', 1))
        genres = ' '.join([g.replace(' ', '') for g in row.get('genre_names', [])] * weights.get('genres', 1))
        director = ' '.join([row.get('director', '').replace(' ', '')] * weights.get('director', 1))
        cast = ' '.join([c.replace(' ', '') for c in row.get('top_cast', [])] * weights.get('cast', 1))
        keywords = ' '.join([k.replace(' ', '') for k in row.get('keyword_names', [])] * weights.get('keywords', 1))
        return f"{title} {row.get('overview','')} {genres} {director} {cast} {keywords}"

    movies_df_filtered['combined_features'] = movies_df_filtered.apply(combine_features, axis=1)

    tfidf = TfidfVectorizer(max_features=config.MAX_FEATURES, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df_filtered['combined_features'])
    
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    movie_indices = {movie_id: i for i, movie_id in enumerate(movies_df_filtered['id'])}

    with open(config.FILES['tfidf_matrix'], 'wb') as f: pickle.dump(tfidf_matrix, f)
    with open(config.FILES['cosine_sim_matrix'], 'wb') as f: pickle.dump(cosine_sim_matrix, f)
    with open(config.FILES['movie_indices'], 'wb') as f: pickle.dump(movie_indices, f)
    
    print(f"Content models created and saved. TF-IDF shape: {tfidf_matrix.shape}")
    
    movies_df_filtered.to_csv(config.FILES['movies_clean'], index=False)
    ratings_df.to_csv(config.FILES['ratings_clean'], index=False)
    print("Saved final cleaned and filtered movies_clean.csv and ratings_clean_filtered.csv")

def step_5_create_production_collaborative_model(ratings_df):
    """Creates and saves the main collaborative model for production use."""
    print("\n[STEP 5/5] Creating production collaborative filtering model...")
    
    pivot_table = ratings_df.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    user_movie_matrix = csr_matrix(pivot_table.values)
    
    user_indices = {user_id: i for i, user_id in enumerate(pivot_table.index)}
    movie_indices_collab = {movie_id: i for i, movie_id in enumerate(pivot_table.columns)}
    
    svd = TruncatedSVD(n_components=config.N_FACTORS, random_state=42)
    svd.fit(user_movie_matrix)
    
    model_data = {
        'model': svd,
        'user_movie_matrix': user_movie_matrix,
        'movie_indices': movie_indices_collab,
    }
    
    with open(config.FILES['collaborative_model'], 'wb') as f: pickle.dump(model_data, f)
    with open(config.FILES['user_indices'], 'wb') as f: pickle.dump(user_indices, f)
    print("Production collaborative model and user indices saved.")

def main():
    """Main preprocessing pipeline."""
    start_time = time.time()
    print("="*60)
    print("FILMFUSION DATA PREPROCESSING PIPELINE STARTED")
    print("="*60)

    config.PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)

    movies_df = step_1_clean_movies_data()
    movies_df = step_2_merge_additional_data(movies_df)
    ratings_df = step_3_clean_and_filter_ratings(movies_df)
    step_4_create_content_models(movies_df, ratings_df)
    step_5_create_production_collaborative_model(ratings_df)

    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print("="*60)

if __name__ == '__main__':
    main()