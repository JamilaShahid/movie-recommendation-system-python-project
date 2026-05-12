import math
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ─── NaN cleaner ───────────────────────────────────────────
def clean_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj

# ─── App ───────────────────────────────────────────────────
app = FastAPI(title="Movie Recommender API")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Data load karo ────────────────────────────────────────
df = None
tfidf_matrix = None
indices = None

def load_data():
    global df, tfidf_matrix, indices
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)
    print(f"✅ Data loaded: {len(df)} movies")

load_data()

# ─── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "total_movies": len(df)}

@app.get("/recommend")
def recommend(title: str, n: int = 10):
    title_lower = title.lower().strip()

    # Flexible matching
    matched = [(k, v) for k, v in indices.items()
               if title_lower in str(k).lower()]

    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"'{title}' not found. Check spelling."
        )

    movie_key, idx = matched[0]

    # Cosine similarity
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:n+1]

    results = []
    for i, score in scored:
        row = df.iloc[i]
        results.append(clean_nan({
            "title":  str(row.get("title") or row.get("original_title") or "Unknown"),
            "year":   str(row.get("release_date") or "")[:4],
            "score":  round(float(score), 4),
            "genres": str(row.get("genres") or ""),
            "overview": str(row.get("overview") or ""),
        }))

    return {"query": title, "matched_with": movie_key, "recommendations": results}


@app.get("/search")
def search(q: str, limit: int = 8):
    q_lower = q.lower()
    results = [{"title": k} for k in indices
               if q_lower in str(k).lower()][:limit]
    return {"results": results}