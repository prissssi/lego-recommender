import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title="LEGO Set Recommender", page_icon="🧱", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_csv("lego_sets.csv")
    
    # Standardize column names to lowercase to avoid mismatch issues
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Map common variations of column names
    col_mapping = {}
    for col in df.columns:
        if col in ["name", "set_name", "title"]:
            col_mapping[col] = "name"
        elif col in ["theme", "theme_name", "category"]:
            col_mapping[col] = "theme"
        elif col in ["price", "retail_price", "us_retailprice", "cost"]:
            col_mapping[col] = "price"
        elif col in ["num_parts", "parts", "pieces"]:
            col_mapping[col] = "num_parts"
        elif col in ["set_num", "set_id", "id"]:
            col_mapping[col] = "set_num"
            
    df = df.rename(columns=col_mapping)
    
    # Ensure required columns exist
    required = ["name", "theme", "price"]
    for req in required:
        if req not in df.columns:
            st.error(f"Missing required column in CSV: '{req}'. Found columns: {list(df.columns)}")
            st.stop()
            
    if "num_parts" not in df.columns:
        df["num_parts"] = "N/A"
    if "set_num" not in df.columns:
        df["set_num"] = ""

    # Clean data
    df = df.dropna(subset=["name", "theme", "price"])
    df["price"] = df["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    
    df["search_text"] = df["name"].astype(str) + " " + df["theme"].astype(str)
    return df

# UI Header
st.title("🧱 LEGO Set Recommender")
st.write("Enter your interests and maximum budget to find your next build.")

try:
    df = load_data()
except FileNotFoundError:
    st.error("`lego_sets.csv` not found! Make sure it is uploaded in the same folder as `app.py`.")
    st.stop()

# User Inputs
col1, col2 = st.columns([2, 1])
with col1:
    interest = st.text_input("What are you interested in?", placeholder="e.g., Star Wars, space, cars, castle, flowers...")
with col2:
    max_price = float(df["price"].max()) if not df.empty else 1000.0
    budget = st.number_input("Max Budget ($)", min_value=1.0, max_value=max(max_price, 1000.0), value=100.0, step=5.0)

num_results = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Recommendation Engine
if st.button("Find LEGO Sets", type="primary", use_container_width=True):
    affordable = df[df["price"] <= budget].copy()
    
    if affordable.empty:
        st.warning(f"No LEGO sets found under ${budget:.2f}. Try increasing your budget!")
    else:
        if not interest.strip():
            st.subheader(f"Top sets under ${budget:.2f}")
            results = affordable.sort_values(by="price", ascending=False).head(num_results)
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(affordable["search_text"])
            user_vec = vectorizer.transform([interest])
            
            affordable["match_score"] = cosine_similarity(user_vec, tfidf_matrix).flatten()
            results = affordable[affordable["match_score"] > 0].sort_values(
                by=["match_score", "price"], ascending=[False, False]
            ).head(num_results)
            
            if results.empty:
                st.info(f"No exact keyword matches for '{interest}' under ${budget:.2f}. Showing top sets in budget instead:")
                results = affordable.sort_values(by="price", ascending=False).head(num_results)
            else:
                st.subheader("Recommended For You")

        # Display Results as Clean Cards
        for _, row in results.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    set_label = f" ({row['set_num']})" if str(row['set_num']).strip() else ""
                    st.markdown(f"**{row['name']}**{set_label}")
                    st.caption(f"Theme: {row['theme']}  |  Pieces: {row['num_parts']}")
                with c2:
                    st.markdown(f"### ${row['price']:.2f}")
