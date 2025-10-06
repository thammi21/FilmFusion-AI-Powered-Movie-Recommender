# hybrid_recommender.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import ast
import config
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from utils import store_user_rating, get_user_ratings, safe_float_conversion, safe_int_conversion

class HybridRecommender:
    def __init__(self, content_weight=None, collaborative_weight=None):
        self.base_content_weight = content_weight or 0.4
        self.base_collaborative_weight = collaborative_weight or 0.6
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.movies = None
        self.loaded = False
        self.cold_start_movies = [
            {"id": 550, "title": "Fight Club", "genres": ["Drama", "Thriller"]},
            {"id": 238, "title": "The Godfather", "genres": ["Crime", "Drama"]},
            {"id": 680, "title": "Pulp Fiction", "genres": ["Crime", "Drama"]},
            {"id": 155, "title": "The Dark Knight", "genres": ["Action", "Crime"]},
            {"id": 13, "title": "Forrest Gump", "genres": ["Drama", "Romance"]},
        ]

    def load_models(self):
        """Loads all necessary models for PRODUCTION use."""
        if self.loaded:
            return
        
        print("Loading hybrid system for production...")
        self.content_recommender.load_models()
        self.collaborative_recommender.load_production_model()
        
        if self.content_recommender.loaded and self.collaborative_recommender.loaded:
            self.movies = self.content_recommender.movies
            self.loaded = True
            print("Hybrid system loaded successfully for production.")
        else:
            print("Hybrid system loading failed.")

    def train_for_evaluation(self, train_ratings_df: pd.DataFrame):
        """Loads content models and trains the collaborative model on specific training data."""
        print("Training hybrid system for evaluation...")
        if not self.content_recommender.loaded:
            self.content_recommender.load_models()
        
        self.collaborative_recommender.train(train_ratings_df)
        
        if self.content_recommender.loaded and self.collaborative_recommender.loaded:
            self.movies = self.content_recommender.movies
            self.loaded = True
            print("Hybrid system trained successfully for evaluation.")
        else:
            print("Hybrid system training for evaluation failed.")

    def is_new_user(self, user_id: int) -> bool:
        """Check if user needs onboarding (lowered threshold)."""
        user_ratings = get_user_ratings(user_id)
        if user_ratings and len(user_ratings) >= 2:
            return False
            
        if self.collaborative_recommender.loaded:
            user_profile = self.collaborative_recommender.get_user_profile(user_id)
            if user_profile and user_profile.get('total_ratings', 0) >= 2:
                return False
        return True

    def get_cold_start_movies(self, n_recommendations: int = 15) -> List[Dict]:
        """Get diverse movies for new user onboarding."""
        return self.content_recommender.get_top_rated_movies(n_recommendations)

    def get_personalized_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 20,
        include_explanations: bool = True
    ) -> List[Dict]:
        if not self.loaded:
            raise RuntimeError("Recommender system is not loaded or trained.")

        user_ratings = get_user_ratings(user_id)
        user_profile = self.collaborative_recommender.get_user_profile(user_id)
        total_ratings = (user_profile.get('total_ratings', 0) if user_profile else 0) + len(user_ratings)

        if total_ratings < 3:
            return self.get_cold_start_movies(n_recommendations)

        content_w, collab_w = self._calculate_dynamic_weights(total_ratings)
        
        content_recs = self.get_user_aware_content_recommendations(user_id, n_recommendations * 2)
        collaborative_recs = self.collaborative_recommender.get_user_recommendations(user_id, n_recommendations * 2)
        
        hybrid_recs = self._combine_recommendations_improved(content_recs, collaborative_recs, content_w, collab_w, user_ratings)
        diversified_recs = self._diversify_by_genre(hybrid_recs, n_recommendations)
        
        if include_explanations:
            return self._add_explanations(diversified_recs, user_ratings, user_profile)
        
        return diversified_recs

    def get_similar_movies(self, movie_id: int, n_recommendations: int = 20) -> List[Dict]:
        if not self.loaded:
            raise RuntimeError("Recommender system is not loaded or trained.")
        
        content_similar = self.content_recommender.get_movie_recommendations(movie_id, n_recommendations * 2)
        collab_similar = self.collaborative_recommender.get_item_based_recommendations(movie_id, n_recommendations)
        
        content_similar = self._ensure_dict_format(content_similar)
        collab_similar = self._ensure_dict_format(collab_similar)
        
        if collab_similar:
            hybrid_recs = self._combine_recommendations_improved(content_similar, collab_similar, 0.6, 0.4, {})
        else:
            hybrid_recs = content_similar
        
        return hybrid_recs[:n_recommendations]

    def get_user_aware_content_recommendations(self, user_id: int, n: int) -> List[Dict]:
        """Get content-based recommendations considering user's rating history."""
        user_ratings = get_user_ratings(user_id)
        high_rated_movies = [mid for mid, info in user_ratings.items() if info['rating'] >= 4.0]

        if not high_rated_movies:
            return self.content_recommender.get_top_rated_movies(n)

        # Base recommendations on the most recent highly rated movie
        latest_movie_id = max(high_rated_movies, key=lambda mid: user_ratings[mid].get('timestamp', 0))
        recs = self.content_recommender.get_movie_recommendations(latest_movie_id, n)
        
        # Filter out movies already rated
        return [rec for rec in recs if rec['id'] not in user_ratings]

    def _calculate_dynamic_weights(self, total_ratings: int) -> Tuple[float, float]:
        if total_ratings >= 15: return 0.2, 0.8
        elif total_ratings >= 8: return 0.3, 0.7
        elif total_ratings >= 5: return 0.4, 0.6
        else: return 0.45, 0.55

    def _combine_recommendations_improved(self, content_recs, collaborative_recs, content_w, collab_w, user_ratings):
        combined = {}
        
        for rec in content_recs:
            if rec['id'] not in user_ratings:
                combined[rec['id']] = {'rec': rec, 'score': rec.get('similarity_score', 0) * content_w, 'source': ['Content']}
        
        for rec in collaborative_recs:
            if rec['id'] not in user_ratings:
                if rec['id'] in combined:
                    combined[rec['id']]['score'] += rec.get('similarity_score', 0) * collab_w
                    combined[rec['id']]['source'].append('Collaborative')
                else:
                    combined[rec['id']] = {'rec': rec, 'score': rec.get('similarity_score', 0) * collab_w, 'source': ['Collaborative']}
        
        sorted_recs = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        final_recs = []
        for item in sorted_recs:
            rec = item['rec']
            rec['hybrid_score'] = item['score']
            rec['recommendation_reason'] = f"Hybrid ({' + '.join(item['source'])})"
            final_recs.append(rec)
            
        return final_recs

    def _diversify_by_genre(self, recommendations: List[Dict], n_recommendations: int) -> List[Dict]:
        if len(recommendations) <= n_recommendations:
            return recommendations

        genre_groups = {}
        for rec in recommendations:
            genres = rec.get('genre_names', [])
            primary_genre = genres[0] if genres else 'Unknown'
            if primary_genre not in genre_groups:
                genre_groups[primary_genre] = []
            genre_groups[primary_genre].append(rec)

        diversified = []
        # Round-robin selection from genres
        while len(diversified) < n_recommendations:
            has_changed = False
            for genre in genre_groups:
                if genre_groups[genre]:
                    diversified.append(genre_groups[genre].pop(0))
                    has_changed = True
                if len(diversified) >= n_recommendations:
                    break
            if not has_changed:
                break
        
        return diversified

    def _add_explanations(self, recommendations: List[Dict], user_ratings: Dict, user_profile: Optional[Dict]) -> List[Dict]:
        # You can expand this with more detailed logic
        for rec in recommendations:
            rec['explanation'] = rec.get('recommendation_reason', 'Recommended for you')
        return recommendations

    def _ensure_dict_format(self, recommendations):
        formatted = []
        for rec in recommendations:
            if isinstance(rec, pd.Series):
                formatted.append(rec.to_dict())
            elif isinstance(rec, dict):
                formatted.append(rec)
        return formatted

    def update_user_preferences(self, user_id: int, movie_id: int, rating: float):
        store_user_rating(user_id, movie_id, rating)
        print(f"Updated preferences: User {user_id} rated movie {movie_id}: {rating}/5")
        
    def get_model_statistics(self) -> Dict:
        if not self.loaded: return {}
        collab_stats = self.collaborative_recommender.get_model_statistics()
        return {
            'total_movies': len(self.movies) if self.movies is not None else 0,
            'collaborative_users': collab_stats.get('total_users', 0),
            'collaborative_movies': collab_stats.get('total_movies', 0),
            'model_loaded': self.loaded,
        }