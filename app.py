import math
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

def clean_val(val):
    try:
        if isinstance(val, float) and math.isnan(val):
            return ""
    except:
        pass
    return str(val) if val else ""

# ─── Load & Build Data ─────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Movie database load ho rahi hai... thoda wait karo!")
def load_data():
    # Working TMDB 5000 dataset URL
    url = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/TMDB%20Recommender%20System.csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        return None, None, None, str(e)

    # Find title column
    title_col = None
    for col in ['title', 'movie_title', 'Title', 'name']:
        if col in df.columns:
            title_col = col
            break
    if title_col is None:
        return None, None, None, f"Title column nahi mila. Columns: {df.columns.tolist()}"

    if title_col != 'title':
        df = df.rename(columns={title_col: 'title'})

    # Fill text columns
    for col in ['overview', 'genres', 'tagline', 'keywords']:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            df[col] = ''

    df['tags'] = df['overview'] + ' ' + df['genres'] + ' ' + df['tagline'] + ' ' + df['keywords']

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['tags'])

    # Indices
    df['title'] = df['title'].fillna('Unknown')
    indices = pd.Series(df.index, index=df['title'].str.lower())

    return df, tfidf_matrix, indices, None

df, tfidf_matrix, indices, error = load_data()

# ─── Recommend ─────────────────────────────────────────────
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
            "title": clean_val(row.get("title")),
            "year": clean_val(row.get("release_date", ""))[:4],
            "score": round(float(score), 4),
            "genres": clean_val(row.get("genres", "")),
            "overview": clean_val(row.get("overview", "")),
            "vote": clean_val(row.get("vote_average", "")),
        })
    return movie_key, results

# ─── UI ────────────────────────────────────────────────────
st.title("🎬 Movie Recommendation System")
st.markdown("Apni pasandida movie ka naam likho aur similar movies dekho!")

if error or df is None:
    st.error(f"Data load error: {error}")
else:
    st.caption(f"📽️ Total movies: {len(df)}")

    movie_input = st.text_input("Movie ka naam likho:", placeholder="e.g. Avatar, Inception, Titanic")
    num_results = st.slider("Kitni movies dikhani hain?", 5, 20, 10)

    if st.button("🔍 Recommend Karo", use_container_width=True):
        if not movie_input.strip():
            st.warning("Pehle movie ka naam likho!")
        else:
            matched_key, recommendations = recommend(movie_input, num_results)
            if not recommendations:
                st.error(f"'{movie_input}' nahi mila!")
                q = movie_input.lower()
                suggestions = [k for k in indices.index if q in str(k).lower()][:5]
                if suggestions:
                    st.info("Shayad yeh dhundh rahe hain?")
                    for s in suggestions:
                        st.write(f"• {s.title()}")
            else:
                st.success(f"✅ **'{matched_key}'** ke liye {len(recommendations)} recommendations:")
                st.divider()
                for i, movie in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {movie['title']} ({movie['year']}) ⭐ {movie['vote']} | Match: {movie['score']}"):
                        if movie.get("genres"):
                            st.write(f"**Genres:** {movie['genres']}")
                        if movie.get("overview"):
                            st.write(f"**Overview:** {movie['overview']}")
