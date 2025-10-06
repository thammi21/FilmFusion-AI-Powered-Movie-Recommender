from pydantic import BaseModel, Field
from typing import List, Optional

class MovieResponse(BaseModel):
    id: int = Field(..., description="Movie ID")
    title: str = Field(..., description="Movie title")
    year: Optional[int] = Field(None, description="Release year")
    rating: float = Field(0.0, description="Movie rating", ge=0, le=10)
    genre_names: List[str] = Field(default_factory=list, description="List of genres")
    overview: str = Field("", description="Movie overview")
    runtime: Optional[int] = Field(None, description="Runtime in minutes")
    director: str = Field("", description="Director name")
    similarity_score: float = Field(0.0, description="Similarity score", ge=0, le=1)
    recommendation_reason: str = Field("", description="Why recommended")
    explanation: str = Field("", description="Detailed explanation")
    poster_url: Optional[str] = Field(None, description="Poster image URL")

    class Config:
        # Allow extra fields for flexibility
        extra = "allow"

class RecommendationRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="User ID")
    movie_id: Optional[int] = Field(None, description="Movie ID")
    n_recommendations: int = Field(10, description="Number of recommendations", ge=1, le=100)
    genre: Optional[str] = Field(None, description="Genre filter")
    dynamic_weighting: bool = Field(True, description="Use dynamic weighting")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    user_id: Optional[int] = Field(None, description="User ID")
    n_recommendations: int = Field(10, description="Number of recommendations", ge=1, le=100)

class UserRatingRequest(BaseModel):
    movie_id: int = Field(..., description="Movie ID", ge=1)
    rating: float = Field(..., description="Rating", ge=0.5, le=5.0)

class RecommendationResponse(BaseModel):
    recommendations: List[MovieResponse] = Field(..., description="List of recommended movies")
    total_count: int = Field(..., description="Total number of recommendations")
    recommendation_type: str = Field(..., description="Type of recommendation")
    generated_at: float = Field(..., description="Generation timestamp")

class GenreResponse(BaseModel):
    name: str = Field(..., description="Genre name")
    count: int = Field(0, description="Number of movies in this genre", ge=0)

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    status_code: int = Field(..., description="HTTP status code")