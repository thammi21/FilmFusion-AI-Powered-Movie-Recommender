# collaborative.py

import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Optional
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import config

class CollaborativeRecommender:
    def __init__(self):
        self.model: Optional[TruncatedSVD] = None
        self.user_movie_matrix: Optional[csr_matrix] = None
        self.user_indices: Optional[Dict[int, int]] = None
        self.movie_indices: Optional[Dict[int, int]] = None
        self.inv_movie_indices: Optional[Dict[int, int]] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings: Optional[pd.DataFrame] = None
        self.loaded = False

    def _load_helper_data(self):
        """Loads movies and ratings dataframes if they don't exist."""
        if self.movies_df is None:
            self.movies_df = pd.read_csv(config.FILES['movies_clean'])
        if self.ratings is None:
            self.ratings = pd.read_csv(config.FILES['ratings_clean'])

    def load_production_model(self):
        """Loads the pre-trained collaborative model for production use."""
        try:
            with open(config.FILES['collaborative_model'], 'rb') as f:
                model_data = pickle.load(f)
            with open(config.FILES['user_indices'], 'rb') as f:
                self.user_indices = pickle.load(f)

            self.model = model_data['model']
            self.user_movie_matrix = model_data['user_movie_matrix']
            self.movie_indices = model_data['movie_indices']
            self.inv_movie_indices = {i: movie_id for movie_id, i in self.movie_indices.items()}
            self._load_helper_data()
            self.loaded = True
            print("Collaborative production model loaded successfully.")
        except Exception as e:
            print(f"Error loading collaborative production model: {e}")
            self.loaded = False

    def train(self, ratings_df: pd.DataFrame):
        """Trains the SVD model on a given ratings dataframe."""
        print("Training collaborative model on provided data...")
        try:
            self.ratings = ratings_df # Use the passed ratings for this instance
            pivot_table = self.ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
            self.user_movie_matrix = csr_matrix(pivot_table.values)

            self.user_indices = {user_id: i for i, user_id in enumerate(pivot_table.index)}
            self.movie_indices = {movie_id: i for i, movie_id in enumerate(pivot_table.columns)}
            self.inv_movie_indices = {i: movie_id for movie_id, i in self.movie_indices.items()}

            n_components = min(config.N_FACTORS, self.user_movie_matrix.shape[1] - 1, 100)
            self.model = TruncatedSVD(n_components=n_components, random_state=42)
            self.model.fit(self.user_movie_matrix)
            
            self._load_helper_data()
            self.loaded = True
            print(f"Collaborative model trained with {n_components} components.")
        except Exception as e:
            print(f"Error during collaborative model training: {e}")
            self.loaded = False

    def get_user_recommendations(self, user_id: int, n_recommendations: int = 20) -> List[Dict]:
        """Get recommendations for a specific user."""
        if not self.loaded or user_id not in self.user_indices:
            return []

        try:
            user_idx = self.user_indices[user_id]
            user_vector = self.model.transform(self.user_movie_matrix[user_idx])
            scores = user_vector.dot(self.model.components_).flatten()

            rated_movie_indices = self.user_movie_matrix[user_idx].nonzero()[1]
            scores[rated_movie_indices] = -np.inf

            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            
            recommendations = []
            for idx in top_indices:
                movie_id = self.inv_movie_indices.get(idx)
                if movie_id:
                    movie_data = self.movies_df[self.movies_df['id'] == movie_id]
                    if not movie_data.empty:
                        rec_dict = movie_data.iloc[0].to_dict()
                        rec_dict['similarity_score'] = scores[idx]
                        rec_dict['recommendation_reason'] = 'Collaborative Filtering'
                        recommendations.append(rec_dict)
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {e}")
            return []

    def get_item_based_recommendations(self, movie_id: int, n_recommendations: int = 20) -> List[Dict]:
        """Get item-based recommendations for a given movie."""
        if not self.loaded or movie_id not in self.movie_indices:
            return []
        
        try:
            movie_idx = self.movie_indices[movie_id]
            movie_vector = self.model.components_[movie_idx].reshape(1, -1)
            similarities = np.dot(movie_vector, self.model.components_.T).flatten()
            
            similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
            
            recommendations = []
            for idx in similar_indices:
                rec_movie_id = self.inv_movie_indices.get(idx)
                if rec_movie_id:
                    movie_data = self.movies_df[self.movies_df['id'] == rec_movie_id]
                    if not movie_data.empty:
                        movie_dict = movie_data.iloc[0].to_dict()
                        movie_dict['similarity_score'] = float(similarities[idx])
                        movie_dict['recommendation_reason'] = 'Item-based collaborative'
                        recommendations.append(movie_dict)
            return recommendations
        except Exception as e:
            print(f"Error getting item-based recommendations: {e}")
            return []

    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get comprehensive user profile information."""
        if not self.loaded or user_id not in self.user_indices:
            return None
            
        try:
            user_idx = self.user_indices[user_id]
            user_ratings = self.user_movie_matrix[user_idx].toarray().flatten()
            
            non_zero_ratings = user_ratings[user_ratings > 0]
            if len(non_zero_ratings) == 0:
                return {'user_id': user_id, 'total_ratings': 0}
                
            profile = {
                'user_id': user_id,
                'total_ratings': len(non_zero_ratings),
                'average_rating': float(np.mean(non_zero_ratings)),
                'rating_std': float(np.std(non_zero_ratings)),
                'min_rating': float(np.min(non_zero_ratings)),
                'max_rating': float(np.max(non_zero_ratings)),
            }
            return profile
        except Exception as e:
            print(f"Error getting user profile: {e}")
            return None
            
    def get_model_statistics(self) -> Dict:
        """Get detailed model statistics."""
        if not self.loaded: return {}
        
        try:
            return {
                'total_users': len(self.user_indices),
                'total_movies': len(self.movie_indices),
                'matrix_shape': self.user_movie_matrix.shape,
                'matrix_density': self.user_movie_matrix.nnz / (self.user_movie_matrix.shape[0] * self.user_movie_matrix.shape[1]),
                'n_components': self.model.n_components if self.model else 0,
                'explained_variance_ratio': self.model.explained_variance_ratio_.sum() if self.model and hasattr(self.model, 'explained_variance_ratio_') else 0,
            }
        except Exception as e:
            print(f"Error getting model statistics: {e}")
            return {}