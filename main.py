import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import joblib 
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="🎬 Movie-Informed Book Recs", layout="wide")
st.title("🎬 Movie Preferences → 📚 Book Recommendations")

# --- CONSTANT DEFINITION ---
LOCAL_MODEL_PATH = './local_sentence_transformer_model'
GENRE_EMBEDDINGS_PATH = 'genre_embeddings.npy' 
USER_EMBEDDINGS_PATH = 'user_embeddings.npy' 
NN_GENRE_MODEL_PATH = 'nn_model_genre.joblib' 
NN_USER_MODEL_PATH = 'nn_model_user.joblib'   
BOOK_DATA_PATH = 'datasets/data.csv'
USER_PROFILES_PATH = 'datasets/user_profiles.csv' 


# --- HELPER FUNCTION ---
def map_age_to_bucket(age: int) -> str:
    """Maps a raw age value to one of the defined age buckets."""
    if age < 18: return 'Under 18'
    elif 18 <= age <= 24: return '18-24'
    elif 25 <= age <= 34: return '25-34'
    elif 35 <= age <= 44: return '35-44'
    elif 45 <= age <= 49: return '45-49'
    elif 50 <= age <= 55: return '50-55'
    else: return '56+'

# --- DATA LOADING (NON-CACHED) ---
def load_and_preprocess_data(_status_placeholder): 
    """Loads, cleans, and prepares books data and user profiles."""
    _status_placeholder.text("1/3: Loading and Preprocessing Data Files...") 
    
    try:
        # 1. Load Books Data (df)
        df_books = pd.read_csv(BOOK_DATA_PATH)
        df_books['genres_string'] = df_books['genres'].fillna('').apply(
            lambda x: ' '.join(x) if isinstance(x, list) else x
        )
        
        # 2. Load User Profiles Data
        user_profiles_df = pd.read_csv(USER_PROFILES_PATH)
        
        def safe_literal_eval(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
            except:
                return x
                
        user_profiles_df['genres'] = user_profiles_df['genres'].apply(safe_literal_eval)
        
        # Prepare features for User Similarity Model
        user_profiles_df['age_bucket'] = user_profiles_df['age'].apply(map_age_to_bucket)
        user_profiles_df['feature_string'] = (
            user_profiles_df['gender'] + ' ' +
            user_profiles_df['occupation_name'] + ' ' +
            user_profiles_df['age_bucket'] + ' ' +
            user_profiles_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        )
        
        return df_books, user_profiles_df
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}. Please ensure data.csv and movie user files are present.")
        return pd.DataFrame(), pd.DataFrame()


# --- CACHED MODEL INITIALIZATION ---
def initialize_and_train_models(df_data, user_profiles_df, _status_placeholder, _progress_bar_placeholder): 
    """
    Initializes SentenceTransformers and loads pre-trained NN models (or trains them if missing).
    """
    
    # 1. Check for basic prerequisites
    if not os.path.exists(LOCAL_MODEL_PATH) or not os.listdir(LOCAL_MODEL_PATH):
        _status_placeholder.error(f"🚨 Model weights not found at **{LOCAL_MODEL_PATH}**. Please run `python model_setup.py` first.")
        st.stop()
        
    # --- FAST PATH: Load Pre-trained NN Models (The desired path) ---
    if os.path.exists(NN_GENRE_MODEL_PATH) and os.path.exists(NN_USER_MODEL_PATH):
        _status_placeholder.text("2/3: Loading all pre-trained models (Instantaneous)...")
        _progress_bar_placeholder.progress(33)
        
        # Load model instances (needed for new query encoding)
        model_genre = SentenceTransformer(LOCAL_MODEL_PATH)
        model_user = SentenceTransformer(LOCAL_MODEL_PATH)
        
        # Load pre-trained NearestNeighbors models (INSTANT LOAD)
        nn_model_genre = joblib.load(NN_GENRE_MODEL_PATH) 
        nn_model_user = joblib.load(NN_USER_MODEL_PATH)   
        
        _status_placeholder.text("3/3: All models loaded and ready. ✅")
        _progress_bar_placeholder.progress(100)
        st.toast("App startup complete! ✅", icon='⚡')
        return model_genre, nn_model_genre, model_user, nn_model_user

    # --- SLOW PATH (Fallback: If joblib files don't exist, we must train) ---
    if not os.path.exists(GENRE_EMBEDDINGS_PATH) or not os.path.exists(USER_EMBEDDINGS_PATH):
        _status_placeholder.error("🚨 Embeddings files are missing. Please run the setup script to create them.")
        st.stop()

    _status_placeholder.text("2/4: Loading Models and Embeddings (Trained models not found, training required)...")
    _progress_bar_placeholder.progress(25)
    
    # Load Model Instances & Embeddings
    model_genre = SentenceTransformer(LOCAL_MODEL_PATH)
    model_user = SentenceTransformer(LOCAL_MODEL_PATH) 
    genre_embeddings = np.load(GENRE_EMBEDDINGS_PATH)
    user_embeddings = np.load(USER_EMBEDDINGS_PATH)
    K_NEIGHBORS = 50 # Constant for NN training

    # 3. Training and Saving Genre Nearest Neighbors (SLOW STEP)
    _status_placeholder.text("3/4: Training Item NearestNeighbors Model...")
    _progress_bar_placeholder.progress(50)
    nn_model_genre = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine', algorithm='auto')
    nn_model_genre.fit(genre_embeddings)
    joblib.dump(nn_model_genre, NN_GENRE_MODEL_PATH) # <-- SAVE TRAINED MODEL

    # 4. Training and Saving User Nearest Neighbors (SLOW STEP)
    _status_placeholder.text("4/4: Training User NearestNeighbors Model...")
    _progress_bar_placeholder.progress(75)
    nn_model_user = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric='cosine', algorithm='auto')
    nn_model_user.fit(user_embeddings)
    joblib.dump(nn_model_user, NN_USER_MODEL_PATH) # <-- SAVE TRAINED MODEL

    # Finalize
    _progress_bar_placeholder.progress(100)
    _status_placeholder.success("App Ready! ✅ (Trained models saved for next run!)")
    
    return model_genre, nn_model_genre, model_user, nn_model_user

# ==============================================================================
# --- CORE RECOMMENDATION LOGIC (Functions defined separately) ---
# ==============================================================================

def get_ranked_recommendations_for_user(user_id, user_profile_data, df_data, model_genre, nn_model_genre, WEIGHT_GENRE = 0.95, WEIGHT_RATING = 0.05, top_n=10, initial_candidates=50):
    """ Content-Based Re-ranking for a single user. """
    user_profile = user_profile_data[user_profile_data['user_id'] == user_id].iloc[0]
    genre_list = user_profile['genres']
    
    if not isinstance(genre_list, list) or not genre_list: return pd.DataFrame()

    genre_string = ' '.join(genre_list)
    query_embedding = model_genre.encode(genre_string).reshape(1, -1)
    
    distances, indices = nn_model_genre.kneighbors(query_embedding, n_neighbors=initial_candidates)
    candidate_df = df_data.iloc[indices.flatten()].copy()
    candidate_df['genre_distance'] = distances.flatten()
    
    candidate_df['rating'] = pd.to_numeric(candidate_df['rating'], errors='coerce').fillna(0)
    candidate_df['normalized_rating'] = candidate_df['rating'] / 5.0
    candidate_df['normalized_genre_similarity'] = 1 - candidate_df['genre_distance']
    
    candidate_df['combined_relevance_score'] = (WEIGHT_GENRE * candidate_df['normalized_genre_similarity']) + (WEIGHT_RATING * candidate_df['normalized_rating'])
    ranked_recommendations = candidate_df.sort_values(by='combined_relevance_score', ascending=False)
    
    top_recommendations = ranked_recommendations[['title', 'rating', 'combined_relevance_score']].head(top_n)
    top_recommendations['user_id'] = user_id
    top_recommendations['cluster'] = user_profile['cluster']
    
    return top_recommendations

def predict_user_cluster(new_user_age: int, new_user_gender: str, new_user_occupation: str, new_user_genres: list, user_profiles_df: pd.DataFrame, model_user: SentenceTransformer, nn_model_user: NearestNeighbors, k_similar_users: int = 10) -> pd.Series:
    """Predicts the most likely cluster for a new user profile based on K-NN modal cluster."""
    
    new_user_age_bucket = map_age_to_bucket(new_user_age)
    new_user_genre_string = ' '.join(new_user_genres)
    new_user_feature_string = (f"{new_user_gender} {new_user_occupation} {new_user_age_bucket} {new_user_genre_string}")
    
    new_user_embedding = model_user.encode(new_user_feature_string).reshape(1, -1)
    
    distances, indices = nn_model_user.kneighbors(new_user_embedding, n_neighbors=k_similar_users)
    similar_user_indices = indices.flatten()
    similar_users_clusters = user_profiles_df.iloc[similar_user_indices]['cluster']
    cluster_counts = similar_users_clusters.value_counts()
    predicted_cluster = cluster_counts.index[0] if not cluster_counts.empty else None
    
    return pd.Series({'predicted_cluster': predicted_cluster, 'cluster_distribution': cluster_counts.to_dict()})

def get_combined_recommendations(
    new_user_age: int, new_user_gender: str, new_user_occupation: str, new_user_genres: list, 
    user_profiles_df: pd.DataFrame, item_data: pd.DataFrame,
    model_user: SentenceTransformer, nn_model_user: NearestNeighbors,
    model_genre: SentenceTransformer, nn_model_genre: NearestNeighbors,
    top_n_items: int = 10, k_similar_users: int = 3, WEIGHT_GENRE: float = 0.95, WEIGHT_RATING: float = 0.05
):
    """ Master function: CF (Find Users) -> CB (Re-rank) -> Aggregate. """
    
    # --- Part 1: Find K Similar Users & Extract Details (CF Component) ---
    new_user_age_bucket = map_age_to_bucket(new_user_age)
    new_user_genre_string = ' '.join(new_user_genres)
    new_user_feature_string = (f"{new_user_gender} {new_user_occupation} {new_user_age_bucket} {new_user_genre_string}")
    new_user_embedding = model_user.encode(new_user_feature_string).reshape(1, -1)
    
    distances, indices = nn_model_user.kneighbors(new_user_embedding, n_neighbors=k_similar_users)
    similar_user_indices = indices.flatten()
    
    similar_users_details = user_profiles_df.iloc[similar_user_indices][['user_id', 'cluster', 'genres']].reset_index(drop=True)
    similar_user_ids = similar_users_details['user_id'].tolist()
    nearest_users_report = similar_users_details.to_dict('records')

    # --- Part 2: Generate and Aggregate Recommendations ---
    all_recommendations = []
    for user_id in similar_user_ids:
        rec_df = get_ranked_recommendations_for_user(
            user_id=user_id, user_profile_data=user_profiles_df, df_data=item_data,                 
            model_genre=model_genre, nn_model_genre=nn_model_genre,
            WEIGHT_GENRE=WEIGHT_GENRE, WEIGHT_RATING=WEIGHT_RATING, top_n=20, initial_candidates=50
        )
        all_recommendations.append(rec_df.drop(columns=['user_id', 'cluster']))

    if not all_recommendations: return pd.DataFrame(), nearest_users_report
    
    final_combined_df = pd.concat(all_recommendations)
    final_combined_df = pd.merge(final_combined_df, item_data[['title', 'genres_string']].drop_duplicates(), on='title', how='left')
    
    # --- Part 3: Aggregate and Re-rank the Final List ---
    final_ranking = final_combined_df.groupby('title').agg(
        avg_combined_relevance_score=('combined_relevance_score', 'mean'),
        max_rating=('rating', 'max'),
        recommendation_count=('title', 'count'), 
        book_genres=('genres_string', 'first') 
    ).reset_index()
    
    final_ranking = final_ranking.sort_values(by=['avg_combined_relevance_score', 'recommendation_count'], ascending=[False, False])
    final_ranking.rename(columns={'max_rating': 'rating'}, inplace=True)
    
    top_recommendations = final_ranking[['title', 'rating', 'avg_combined_relevance_score', 'book_genres']].head(top_n_items)
    
    return top_recommendations, nearest_users_report

# ==============================================================================
# --- APP INITIALIZATION (Runs once at the start) ---
# ==============================================================================

st.sidebar.title("App Status")
initial_loading_spinner = st.empty() 
status_placeholder = st.sidebar.empty()
progress_bar_placeholder = st.sidebar.empty()

# LOAD ALL DATA AND MODELS
with initial_loading_spinner.container():
    with st.spinner("Initializing Data and Models..."):
        df, user_profiles = load_and_preprocess_data(status_placeholder) 

        if df.empty or user_profiles.empty:
            st.stop()

        try:
            model_genre, nn_model_genre, model_user, nn_model_user = initialize_and_train_models(
                df, user_profiles, status_placeholder, progress_bar_placeholder
            )
        except SystemExit:
            st.stop()
            
initial_loading_spinner.empty()
progress_bar_placeholder.empty()
status_placeholder.empty()

# ==============================================================================
# --- SIDEBAR NAVIGATION SETUP ---
# ==============================================================================

st.sidebar.header("Application Views")
selected_view = st.sidebar.radio(
    "Go to",
    ["Data Overview 📊", "New User Profile Recs 🤝", "Existing User Recs 👤"]
)

# ==============================================================================
# --- CONDITIONAL RENDERING ---
# ==============================================================================

if selected_view == "Data Overview 📊":
    st.header("Data Overview: Books and Movie User Profiles")
    st.markdown("Use the tabs below to inspect the datasets used by the recommendation engine.")

    # --- START OF TABBED INTERFACE ---
    tab_books, tab_movies = st.tabs(["📚 Books Dataset", "🎬 Movie User Profiles"])

    with tab_books:
        st.subheader("Book Data Overview")
        st.info(f"Showing first 5 rows of the **{len(df)}** total book items.")
        st.dataframe(df.head())
    
    with tab_movies:
        st.subheader("Movie User Profiles Overview")
        
        try:
            user_profiles_display = pd.read_csv(USER_PROFILES_PATH)
            st.info(f"Showing first 5 rows of the **{len(user_profiles_display)}** total user profiles derived from movie ratings.")
            st.dataframe(user_profiles_display.head())
        except FileNotFoundError:
            st.warning(f"Could not load display data from {USER_PROFILES_PATH}. Please ensure the file exists.")
    

elif selected_view == "Existing User Recs 👤":
    st.header("Recommendations for Existing Users (Content-Based)")
    st.caption("Select a known user ID to view content-based recommendations based solely on their derived cluster preferences.")

    user_id = st.selectbox("Select User ID", user_profiles['user_id'].tolist())

    if user_id:
        with st.spinner(f"Generating content-based recommendations for User {user_id}..."):
            rec_df = get_ranked_recommendations_for_user(user_id, user_profiles, df, model_genre, nn_model_genre, top_n=10)
            
            st.write(f"Top Recommendations for User ID **{user_id}**:")
            st.dataframe(rec_df[['title', 'rating', 'combined_relevance_score']])


elif selected_view == "New User Profile Recs 🤝":
    st.header("Recommend for a New User Profile (Collaborative + Content)")
    st.caption("Enter a new user's profile to find similar users and aggregate their top-ranked books.")

    with st.form("new_user_form"):
        # Get available genres and occupations for the dropdowns
        all_genres = set(g for sublist in user_profiles['genres'].dropna() for g in sublist)
        occupations = user_profiles['occupation_name'].unique()

        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_age = st.slider("Age", 10, 70, 28)
            new_gender = st.selectbox("Gender", ['M', 'F'], index=0)
        
        with col2:
            # FIX: Cast the result of np.where to a standard Python integer
            default_occ_idx = int(np.where(occupations == 'programmer')[0][0]) if 'programmer' in occupations else 0 
            new_occupation = st.selectbox("Occupation", occupations, index=default_occ_idx)
            new_genres = st.multiselect("Preferred Genres", sorted(list(all_genres)), default=['Sci-Fi', 'Action', 'Thriller'])
        
        with col3:
            K_USERS = st.slider("Number of Similar Users (K)", 1, 10, 3)
            TOP_N = st.slider("Number of Books to Recommend (N)", 1, 20, 5)

        submitted = st.form_submit_button("Get Combined Recommendations")

    # --- Recommendation Logic (Post-Submission) ---
    if submitted:
        if not new_genres:
            st.warning("Please select at least one preferred genre.")
        else:
            with st.spinner(f"Finding {K_USERS} similar users and aggregating recommendations..."):
                recommendations_df, user_context = get_combined_recommendations(
                    new_user_age=new_age, new_user_gender=new_gender, new_user_occupation=new_occupation, new_user_genres=new_genres,
                    user_profiles_df=user_profiles, item_data=df,                      
                    model_user=model_user, nn_model_user=nn_model_user,
                    model_genre=model_genre, nn_model_genre=nn_model_genre,   
                    top_n_items=TOP_N, k_similar_users=K_USERS
                )

            st.subheader("Top Recommended Books 🏆")
            st.info("Recommendations are aggregated from the top content-based suggestions for the most similar users.")
            st.dataframe(recommendations_df)

            st.subheader("Interpretation: Influencing User Profiles")
            
            # Display Nearest User Context
            st.markdown(f"**Nearest {K_USERS} Users (CF Component):**")
            context_table = []
            for user in user_context:
                context_table.append({
                    "User ID": user['user_id'],
                    "Cluster": user['cluster'],
                    "Cluster Genres": ', '.join(user['genres'])
                })
            st.dataframe(pd.DataFrame(context_table))

            # Display the Cluster Prediction
            cluster_prediction = predict_user_cluster(
                new_user_age=new_age, new_user_gender=new_gender, new_user_occupation=new_occupation, new_user_genres=new_genres,
                user_profiles_df=user_profiles, model_user=model_user, nn_model_user=nn_model_user, k_similar_users=K_USERS
            )
            st.markdown(f"**Predicted Profile Cluster:** The new user is most similar to **Cluster {cluster_prediction['predicted_cluster']}**.")