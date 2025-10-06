# 🎬📚 Books and Movies Recommendation System

<div align="center">
  
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://books-and-movies-reccomendation-system.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


[🚀 Live Demo](https://books-and-movies-reccomendation-system.streamlit.app/)

</div>

---

## ✨ Features

🎯 **Hybrid Recommendation Engine**
- **Collaborative Filtering**: Find users with similar tastes and preferences
- **Content-Based Filtering**: Analyze genres, ratings, and content similarity
- **User Profiling**: Demographic and preference-based matching

🧠 **AI-Powered Matching**
- **SentenceTransformers**: Advanced text embedding for semantic similarity
- **K-Nearest Neighbors**: Efficient similarity search and clustering
- **Smart Weighting**: Optimized genre and rating balance

🎨 **Interactive Interface**
- **Real-time Recommendations**: Instant book suggestions
- **User Demographics**: Age, gender, and occupation profiling  
- **Visual Analytics**: Cluster analysis and preference insights
- **Responsive Design**: Works on desktop and mobile

**Technology Stack:**
- **Frontend**: Streamlit (Interactive UI)
- **ML/AI**: SentenceTransformers, scikit-learn
- **Data**: Pandas, NumPy for processing
- **Models**: Pre-trained embeddings + custom NN models

## 🚀 Quick Start

### 🔧 Local Development

```bash
# 1️⃣ Clone the repository
git clone https://github.com/sluskii/books-and-movies-reccomendation-system.git
cd books-and-movies-reccomendation-system

# 2️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -U pip
pip install -r requirements.txt

# 4️⃣ Run the application
streamlit run main.py
```

🌐 **Open your browser** to `http://localhost:8501`
