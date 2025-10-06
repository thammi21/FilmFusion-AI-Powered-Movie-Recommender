FilmFusion: An AI-Powered Hybrid Movie Recommender System
FilmFusion is a comprehensive, AI-powered movie recommendation system designed to provide personalized and relevant movie suggestions. It leverages a sophisticated hybrid filtering approach, combining the strengths of Content-Based Filtering and Collaborative Filtering to handle a variety of scenarios, from new users (cold start) to seasoned movie enthusiasts.

The entire system is exposed via a robust FastAPI backend, making it scalable, fast, and ready for integration with frontend applications.

Key Features
Hybrid Recommendation Engine: Dynamically blends content-based and collaborative filtering. The system intelligently adjusts its recommendation strategy based on the user's activity, relying more on collaborative data as the user rates more movies.

Content-Based Filtering: Analyzes movie metadata such as genres, overview, director, cast, and keywords. It uses a TF-IDF vectorizer and Cosine Similarity to find movies with similar attributes.

Collaborative Filtering: Employs matrix factorization (Truncated SVD) on the user-item rating matrix to discover latent user preferences and recommend movies enjoyed by similar users.

Robust Cold Start Handling: Provides meaningful recommendations for new users by suggesting a curated list of popular and critically acclaimed movies to quickly gather initial preferences.

RESTful API: A modern and efficient backend built with FastAPI and Pydantic for clear data validation, automatic documentation, and high performance.

Data-Driven Pipeline: Includes a full preprocessing pipeline to clean, merge, and transform raw movie data into feature-rich, model-ready datasets.

Offline Evaluation Framework: A rigorous evaluation script to measure the system's performance using standard industry metrics like Precision@k, Recall@k, and nDCG@k.

Tech Stack
Backend: Python, FastAPI, Uvicorn

Machine Learning: Scikit-learn, Pandas, NumPy

Data Handling: CSV, Pickle

Environment: Virtualenv (venv)

Getting Started
Follow these instructions to get a local copy up and running for development and testing purposes.

Prerequisites
Python 3.8+

pip and virtualenv

Installation & Setup
Clone the repository:

git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/FilmFusion-AI-Powered-Movie-Recommender.git
cd FilmFusion-AI-Powered-Movie-Recommender

Create and activate a virtual environment:

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

Download the Data:
This project uses the "Movies Dataset" from Kaggle. Please download it and place the following files into the data/raw/ directory:

movies_metadata.csv

ratings.csv

credits.csv

keywords.csv

Run the Preprocessing Pipeline:
This is a crucial one-time step that cleans the data and trains your production ML models.

python run_preprocessing.py

This will populate the data/processed/ and data/models/ directories.

Launch the API Server:

uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000. You can access the interactive Swagger UI documentation at http://1227.0.0.1:8000/docs.

Project Structure
.
├── data/
│   ├── raw/          # Raw Kaggle CSV files
│   ├── processed/    # Cleaned, final datasets
│   └── models/       # Trained and saved ML models (.pkl)
├── frontend/
│   ├── static/
│   └── templates/
├── .gitignore
├── collaborative.py  # Collaborative Filtering model logic
├── content_based.py  # Content-Based Filtering model logic
├── hybrid_recommender.py # Core hybrid model logic
├── main.py           # FastAPI application entry point
├── run_preprocessing.py # Single script for data pipeline
├── requirements.txt  # Project dependencies
└── README.md

API Endpoints
The system exposes several endpoints for interacting with the recommender. Key endpoints include:

GET /api/recommendations/personalized/{user_id}: Get personalized recommendations for a user.

GET /api/movies/{movie_id}/similar: Find movies similar to a given movie.

POST /api/users/{user_id}/rate: Submit a new movie rating for a user.

GET /api/movies/search?query=...: Search for movies by title.

For a full list and to test them live, visit the /docs endpoint after starting the server.