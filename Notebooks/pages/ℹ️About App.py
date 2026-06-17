import streamlit as st
import pandas as pd
import joblib 
import os

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


</style>
""", unsafe_allow_html=True)

#Displaying image on sidebar
BASE_DIR = Path(__file__).resolve().parent.parent
img_path = BASE_DIR / "images" / "book_image.gif"

with st.sidebar:
    st.sidebar.image(img_path)

# Page Configuration
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="centered"
)
st.title("📚 Book Recommender System")
st.write("")
st.markdown("""
## ℹ️ Introduction

This Book Recommender helps you discover new books based on your interests.

You can:
- 📖 Find books similar to a selected book
- 👤 Discover books based on authors you like
- ⭐ Explore recommendations based on your mood

The system uses machine learning techniques to understand book content and recommend similar titles based on meaning, not just keywords.

## 🌐 Data Sources
- Open Library API (book metadata, subjects, covers)
- Goodreads (book metadata with book descriptions via web scraping)
            
## ⚙️ Technologies Used
- 🐍 Python
- 🎈 Streamlit
- 🤖 NLP (Sentence Embeddings)
- 📊 Pandas & NumPy
- 🔍 Scikit-learn (Cosine Similarity)

---         
Built with ❤️ using Streamlit and NLP.
""")
