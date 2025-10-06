```markdown
# 🎬 FilmFusion: AI-Powered Movie Recommender

FilmFusion is an **AI-driven hybrid movie recommendation system** that combines **Collaborative Filtering** and **Content-Based Filtering** to deliver personalized movie suggestions based on user preferences and movie metadata.

---

## 🚀 Features
- Hybrid Recommendation System: collaborative + content-based filtering  
- SVD, item-based similarity, TF-IDF on movie metadata  
- Optimized for performance on mid-range GPUs/CPUs  
- Simple interactive frontend

---

## 🏗️ Project Structure
```

Movie_Recommender_Sys/
├── collaborative.py
├── content_based.py
├── hybrid_recommender.py
├── run_preprocessing.py
├── models.py
├── utils.py
├── config.py
├── main.py
├── frontend/
│ ├── templates/
│ │ └── index.html
│ └── static/
│ └── app.js
└── requirements.txt

````

---

## 📦 Installation

```bash
git clone https://github.com/thammi21/FilmFusion-AI-Powered-Movie-Recommender.git
cd FilmFusion-AI-Powered-Movie-Recommender

python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
````

---

## 🧩 Dataset

This project uses the **MovieLens dataset**: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

Place CSV files in:

```
data/raw/
├── movies_metadata.csv
├── ratings.csv
├── links.csv
└── credits.csv
```

Preprocess:

```bash
python run_preprocessing.py
```

---

## ▶️ Run the App

```bash
python main.py
```

Access the app at: [http://localhost:5000](http://localhost:5000)

---

## 🧠 Tech Stack

* Python, Pandas, NumPy, Scikit-learn, NLTK
* Flask for frontend
* Hybrid recommender: SVD, TF-IDF, Cosine Similarity

---

## 🧑‍💻 Author

**Mohammed Thameem**
[GitHub Profile](https://github.com/thammi21)

```
```
