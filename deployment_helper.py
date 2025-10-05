import streamlit as st
import os

st.set_page_config(page_title="🎬 Movie-Informed Book Recs", layout="wide")
st.title("🎬 Movie Preferences → 📚 Book Recommendations")

# Check if we're in deployment mode
DEPLOYMENT_MODE = os.getenv('STREAMLIT_DEPLOYMENT', 'cloud')

if DEPLOYMENT_MODE == 'cloud':
    st.error("🌐 **Deployment Configuration Required**")
    st.markdown("""
    This app requires model files that are too large for direct deployment.
    
    ## 🛠️ **Quick Fix Options:**
    
    ### Option 1: Use the Deployment-Ready Version
    1. Change your main file in Streamlit Cloud to: **`main_deployment.py`**
    2. Add these secrets in Streamlit Cloud:
       ```toml
       MODEL_FILES_URL = "https://github.com/sluskii/books-and-movies-reccomendation-system/releases/download/v1.0/model_files.tar.gz"
       STREAMLIT_DEPLOYMENT = "cloud"
       ```
    3. Create a GitHub release with the model files
    
    ### Option 2: Use Hugging Face Models (Recommended)
    - Replace local models with online Hugging Face models
    - No file size limitations
    - Automatic downloads
    
    ### Option 3: Use Git LFS
    - Track large files with Git Large File Storage
    - Push model files to repository
    
    ## 📋 **Current Status:**
    - ❌ Model files not found in deployment
    - ❌ App cannot function without models
    - ✅ Data files are available
    
    ## 💡 **Need Help?**
    Check the `DEPLOYMENT_GUIDE.md` file in the repository for detailed instructions.
    """)
    
    st.info("🔧 **For Developers**: Set environment variable `STREAMLIT_DEPLOYMENT=local` to run locally.")
    
else:
    # This would be the local development path
    st.success("✅ **Local Development Mode**")
    st.info("This appears to be running locally. The full app should work here!")
    
    # Show a simple interface for local testing
    st.markdown("---")
    st.subheader("📁 File Status Check")
    
    files_to_check = [
        './local_sentence_transformer_model',
        'genre_embeddings.npy',
        'user_embeddings.npy', 
        'nn_model_genre.joblib',
        'nn_model_user.joblib',
        'datasets/data.csv',
        'datasets/user_profiles.csv'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            st.success(f"✅ {file_path}")
        else:
            st.error(f"❌ {file_path}")
    
    st.markdown("---")
    st.info("If all files are present, switch back to `main.py` to run the full application.")