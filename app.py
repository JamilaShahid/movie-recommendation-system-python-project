import math
import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ─── NaN cleaner ───────────────────────────────────────────
def clean_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    return obj

# ─── Data Load ─────────────────────────────────────────────
@st.cache_resource
def load_data():
    with open("df.pkl", "rb") as f:
        df = pickle.load(f)
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)
    return df, tfidf_matrix, indices

df, tfidf_matrix, indices = load_data()

# ─── Recommend Function ────────────────────────────────────
def recommend(title, n=10):
    title_lower = title.lower().strip()
    matched = [(k, v) for k, v in indices.items()
               if title_lower in str(k).lower()]
    if not matched:
        return None, []
    movie_key, idx = matched[0]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:n+1]
    results = []
    for i, score in scored:
        row = df.iloc[i]
        results.append(clean_nan({
            "title": str(row.get("title") or row.get("original_title") or "Unknown"),
            "year": str(row.get("release_date") or "")[:4],
            "score": round(float(score), 4),
            "genres": str(row.get("genres") or ""),
            "overview": str(row.get("overview") or ""),
        }))
    return movie_key, results

# ─── UI ────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown("Apni pasandida movie ka naam likho aur similar movies dekhو!")

movie_input = st.text_input("Movie ka naam likhو:", placeholder="e.g. Inception, Avatar, Titanic")

num_results = st.slider("Kitni movies dikhani hain?", 5, 20, 10)

if st.button("🔍 Recommend Karo", use_container_width=True):
    if movie_input.strip() == "":
        st.warning("Pehle movie ka naam likho!")
    else:
        with st.spinner("Searching..."):
            matched_key, recommendations = recommend(movie_input, num_results)

        if not recommendations:
            st.error(f"'{movie_input}' nahi mila. Spelling check karo!")
        else:
            st.success(f"✅ '{matched_key}' ke liye {len(recommendations)} recommendations:")
            for i, movie in enumerate(recommendations, 1):
                with st.expander(f"{i}. {movie['title']} ({movie['year']}) — Score: {movie['score']}"):
                    if movie.get("genres"):
                        st.write(f"**Genres:** {movie['genres']}")
                    if movie.get("overview"):
                        st.write(f"**Overview:** {movie['overview']}")

st.markdown("---")
st.caption(f"Total movies in database: {len(df)}")
