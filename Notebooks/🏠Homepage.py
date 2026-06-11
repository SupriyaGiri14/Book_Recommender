import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# CACHE MODEL LOADING (IMPORTANT)
# -----------------------------
@st.cache_resource
def load_model():
    with open("pickles/full_recommender.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

df_final = model["df_final"].reset_index(drop=True)
X = model["X"]
author_embeddings = model["author_embeddings"]
author_to_idx = model["author_to_idx"]
similarity_matrix = model["similarity_matrix"]

# -----------------------------
# FAST LOOKUP MAPS (IMPORTANT OPTIMIZATION)
# -----------------------------
title_to_idx = {title: i for i, title in enumerate(df_final["title"])}
author_group = df_final.groupby("author")

df_final["author"] = df_final["author"].astype(str).str.strip()

# -----------------------------
# GLOBAL CSS (ONLY ONCE)
# -----------------------------
st.markdown("""
<style>
            
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1f2937;  /* dark gray-blue */
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white;
}

.stApp {
    background-color: #fff9dc !important;
}

/* Main content block (important override) */
[data-testid="stAppViewContainer"] {
    background-color: #fff9dc !important;
}

/* Inner block container */
.block-container {
    background-color: #fff9dc !important;
}
.book-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    margin-bottom: 20px;
    transition: transform 0.2s;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.book-card:hover {
    transform: scale(1.05);
}

.book-card img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    border-radius: 8px;
}

.book-title {
    font-weight: bold;
    margin-top: 10px;
    font-size: 0.9em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.badge {
    background-color: #f1f1f1;
    padding: 5px;
    border-radius: 5px;
    margin-top: 5px;
    font-size: 0.8em;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# BOOK RECOMMENDER (FAST)
# -----------------------------
def get_book_idx(title):
    return title_to_idx.get(title, None)


def recommend_books(book_idx, top_n=10):

    cluster = df_final.loc[book_idx, "cluster"]

    candidates = np.where(df_final["cluster"].values == cluster)[0]
    candidates = candidates[candidates != book_idx]

    sims = cosine_similarity(
        X[book_idx].reshape(1, -1),
        X[candidates]
    )[0]

    top_idx = np.argsort(sims)[::-1][:top_n]
    selected = candidates[top_idx]

    return df_final.iloc[selected][
        ["title", "author", "image", "rating", "author_link"]
    ]

# -----------------------------
# AUTHOR RECOMMENDER
# -----------------------------
def get_similar_authors(author_name, top_k=5):

    if author_name not in author_to_idx:
        return []

    idx = author_to_idx[author_name]

    scores = np.asarray(similarity_matrix[idx]).ravel()

    top_idx = np.argsort(scores)[::-1]
    top_idx = top_idx[top_idx != idx][:top_k]

    return list(author_embeddings.index[top_idx])


def get_top_books(author, n=3):
    if author not in author_group.groups:
        return []

    temp = author_group.get_group(author)

    return (
        temp.sort_values("rating", ascending=False)["title"]
        .drop_duplicates()
        .head(n)
        .tolist()
    )

# -----------------------------
# UI
# -----------------------------
st.title("📚 Book Recommender System")

# -----------------------------
# BOOK MODE
# -----------------------------
book_title = st.selectbox("Select a book:", df_final["title"].values)
top_n = st.slider("Number of recommendations", 3, 10, 6)

if st.button("Recommend Books"):

    book_idx = get_book_idx(book_title)

    if book_idx is None:
        st.error("Book not found")
    else:
        results = recommend_books(book_idx, top_n)

        st.subheader("📚 Recommended Books")

        cols = st.columns(3)

        for i, (_, row) in enumerate(results.iterrows()):
            col = cols[i % 3]

            img = row.get("image", "https://via.placeholder.com/200x300")
            title = row.get("title", "Unknown")
            author = row.get("author", "Unknown")
            rating = row.get("rating", "N/A")
            link = row.get("author_link", "#")

            with col:
                st.markdown(f"""
                <div class="book-card">
                    <a href="{link}" target="_blank" style="text-decoration:none; color:inherit;">
                        <img src="{img}">
                        <div class="book-title">📖 {title}</div>
                        <div class="badge">⭐ {rating}</div>
                    </a>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------
# AUTHOR MODE
# -----------------------------
author = st.selectbox(
    "Select an author:",
    sorted(df_final["author"].dropna().unique())
)

top_k = st.slider("Number of similar authors", 1, 5, 3)

if st.button("Find Similar Authors Books"):

    similar_authors = get_similar_authors(author, top_k)

    st.subheader("👨‍💼 Similar Authors & Books")

    for auth in similar_authors:

        books = get_top_books(auth, 3)

        if not books:
            continue

        author_url = df_final[df_final["author"] == auth]["author_link"].iloc[0] \
            if auth in df_final["author"].values else "#"

        st.markdown(f"### ✍️ <a href='{author_url}' target='_blank'>{auth}</a>",
                    unsafe_allow_html=True)

        cols = st.columns(3)

        for i, title in enumerate(books):

            row = df_final[df_final["title"] == title]

            if row.empty:
                continue

            row = row.iloc[0]

            col = cols[i % 3]

            with col:
                st.markdown(f"""
                <div class="book-card">
                    <a href="{row.get('author_link','#')}" target="_blank">
                        <img src="{row.get('image','https://via.placeholder.com/200x300')}">
                        <div class="book-title">📖 {title}</div>
                        <div class="badge">⭐ {row.get('rating','N/A')}</div>
                    </a>
                </div>
                """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.write("---")
#st.sidebar.info("📚 Book Recommender App")