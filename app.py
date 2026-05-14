import math
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ─── NaN cleaner ───────────────────────────────────────────
def clean_val(val):
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val) if val else ""

# ─── Load & Build Data ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading movie database... Please wait!")
def load_data():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/tmdb_5000_movies.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None, None, None

    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    df['tags'] = df['overview'] + ' ' + df['genres'] + ' ' + df['tagline']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])

    indices = pd.Series(df.index, index=df['title'].str.lower())

    return df, tfidf_matrix, indices

df, tfidf_matrix, indices = load_data()

# ─── Recommend Function ─────────────────────────────────────
def recommend(title, n=10):
    title_lower = title.lower().strip()
    matched = [(k, v) for k, v in indices.items()
               if title_lower in str(k).lower()]
    if not matched:
        return None, []
    movie_key, idx = matched[0]
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:n+1]
    results = []
    for i, score in scored:
        row = df.iloc[i]
        results.append({
            "title": clean_val(row.get("title") or row.get("original_title")),
            "year": clean_val(row.get("release_date"))[:4],
            "score": round(float(score), 4),
            "genres": clean_val(row.get("genres")),
            "overview": clean_val(row.get("overview")),
            "vote_average": clean_val(row.get("vote_average")),
        })
    return movie_key, results

# ─── UI ────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown("Apni pasandida movie ka naam likho aur similar movies dekho!")

if df is None:
    st.error("Data load nahi hua. Dobara try karo.")
else:
    st.caption(f"📽️ Total movies in database: {len(df)}")

    movie_input = st.text_input(
        "Movie ka naam likho:",
        placeholder="e.g. Avatar, Inception, Titanic"
    )

    num_results = st.slider("Kitni movies dikhani hain?", 5, 20, 10)

    if st.button("🔍 Recommend Karo", use_container_width=True):
        if movie_input.strip() == "":
            st.warning("Pehle movie ka naam likho!")
        else:
            matched_key, recommendations = recommend(movie_input, num_results)

            if not recommendations:
                st.error(f"'{movie_input}' nahi mila. Spelling check karo!")
                q = movie_input.lower()
                suggestions = [k for k in indices.index if q in str(k).lower()][:5]
                if suggestions:
                    st.info("Kya aap yeh movie dhundh rahe hain?")
                    for s in suggestions:
                        st.write(f"• {s.title()}")
            else:
                st.success(f"✅ **'{matched_key}'** ke liye {len(recommendations)} recommendations:")
                st.divider()
                for i, movie in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {movie['title']} ({movie['year']}) ⭐ {movie['vote_average']} | Score: {movie['score']}"):
                        if movie.get("genres"):
                            st.write(f"**Genres:** {movie['genres']}")
                        if movie.get("overview"):
                            st.write(f"**Overview:** {movie['overview']}")
