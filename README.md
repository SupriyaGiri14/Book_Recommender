# Book Recommender App

Welcome to the **Book Recommender App**, a machine learning project designed to help bookworms discover their next favorite read by uncovering hidden connections between books, authors, and genres.

## 📖 About the Project
This project explores a hybrid recommendation system that transforms thousands of books into meaningful patterns. By leveraging machine learning, the system suggests books based on semantic meaning rather than simple keyword matching, helping readers discover stories they might otherwise miss.

This project was developed by **Supriya Gir** as an Ironhack Machine Learning project in June 2026.

## ⚙️ How It Works
The system utilizes a hybrid approach combining **NLP embeddings** and **clustering**:

*   **Input**: The system ingests book titles, descriptions, and author information.
*   **Processing**: It generates text embeddings to capture semantic depth.
*   **Clustering**: Books are grouped using HDBSCAN.
*   **Ranking**: The system uses cosine similarity to rank and return the top-N similar book recommendations.

## 📊 Data & Methodology

### Dataset
The model is trained on a combined dataset from **Goodreads** and **Openlibrary**, totaling 8,500 books across 12 genres. Key variables include:
*   Title
*   Author
*   Genres
*   Description

### Data Processing Pipeline
*   **Cleaning**: Removed duplicates and handled missing values to ensure high-quality data.
*   **Normalization**: Applied regex to clean titles and standardized text via case folding.
*   **Embeddings**: Used `all-MiniLM-L6-v2` to transform text into 384-dimensional dense vectors.

## 🚀 Recommendation Features
The app provides multiple ways to find your next great read:
*   **By Favorite Book**: Uses book clusters (HDBSCAN) and cosine similarity to find similar titles.
*   **By Favorite Author**: A combination of `TfidfVectorizer` (for word frequency) and `all-MiniLM-L6-v2` (for semantic meaning) to compute cosine similarity between authors.
*   **By Mood**: An additional feature to cater to your current reading preference.

## 🛠️ Challenges Faced
*   Cleaning inconsistent and noisy book metadata.
*   Selecting the most meaningful machine learning model.
*   Handling high-dimensional embeddings.
*   Implementing advanced ML concepts and designing an intuitive user interface.
*   Managing data loss and recovery.

## 📈 Performance
*   **Clustering**: Successfully clustered books with a Silhouette Score of **0.56**.

---
*For more details, please refer to the project documentation: Book Recommender.pdf*
