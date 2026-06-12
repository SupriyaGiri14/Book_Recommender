# Book_Recommender
# Book Recommender App

Welcome to the **Book Recommender App**, a machine learning project designed to help bookworms discover their next favorite read by uncovering hidden connections between books, authors, and genres[cite: 1].

## 📖 About the Project
This project explores a hybrid recommendation system that transforms thousands of books into meaningful patterns[cite: 1]. By leveraging machine learning, the system suggests books based on semantic meaning rather than simple keyword matching, helping readers discover stories they might otherwise miss[cite: 1].

This project was developed by **Supriya Gir** as an Ironhack Machine Learning project in June 2026[cite: 1].

## ⚙️ How It Works
The system utilizes a hybrid approach combining **NLP embeddings** and **clustering**[cite: 1]:

*   **Input**: The system ingests book titles, descriptions, and author information[cite: 1].
*   **Processing**: It generates text embeddings to capture semantic depth[cite: 1].
*   **Clustering**: Books are grouped using HDBSCAN[cite: 1].
*   **Ranking**: The system uses cosine similarity to rank and return the top-N similar book recommendations[cite: 1].

## 📊 Data & Methodology

### Dataset
The model is trained on a combined dataset from **Goodreads** and **Openlibrary**, totaling 8,500 books across 12 genres[cite: 1]. Key variables include:
*   Title[cite: 1]
*   Author[cite: 1]
*   Genres[cite: 1]
*   Description[cite: 1]

### Data Processing Pipeline
*   **Cleaning**: Removed duplicates and handled missing values to ensure high-quality data[cite: 1].
*   **Normalization**: Applied regex to clean titles and standardized text via case folding[cite: 1].
*   **Embeddings**: Used `all-MiniLM-L6-v2` to transform text into 384-dimensional dense vectors[cite: 1].

## 🚀 Recommendation Features
The app provides multiple ways to find your next great read:
*   **By Favorite Book**: Uses book clusters (HDBSCAN) and cosine similarity to find similar titles[cite: 1].
*   **By Favorite Author**: A combination of `TfidfVectorizer` (for word frequency) and `all-MiniLM-L6-v2` (for semantic meaning) to compute cosine similarity between authors[cite: 1].
*   **By Mood**: An additional feature to cater to your current reading preference[cite: 1].

## 🛠️ Challenges Faced
*   Cleaning inconsistent and noisy book metadata[cite: 1].
*   Selecting the most meaningful machine learning model[cite: 1].
*   Handling high-dimensional embeddings[cite: 1].
*   Implementing advanced ML concepts and designing an intuitive user interface[cite: 1].
*   Managing data loss and recovery[cite: 1].

## 📈 Performance
*   **Clustering**: Successfully clustered books with a Silhouette Score of **0.56**[cite: 1].

---
*For more details, please refer to the project documentation: Book Recommender.pdf*
