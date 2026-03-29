import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ', regex=False)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

def fetch_poster(movie_title):
    api_key = "YOUR_API_KEY_HERE"
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"

    response = requests.get(url)
    data = response.json()

    try:
        poster_path = data['results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("<h1 style='text-align: center;'> Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Decide what you'll be watching within seconds</p>", unsafe_allow_html=True)

selected_movie = st.selectbox(
    "Search for a movie:",
    movies['title'].values
)

st.write("")

if st.button("Recommend Movies"):
    results = recommend(selected_movie)

    st.markdown("## Recommended for you")

    cols = st.columns(5)

    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i]:
            poster = fetch_poster(row['title'])

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown("🎬")

            st.markdown(
                f"<p style='text-align:center; font-size:14px;'>{row['title']}</p>",
                unsafe_allow_html=True
            )