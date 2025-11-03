import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import textstat
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords

# === Setup ===
nltk.download('stopwords', quiet=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "quality_model.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(" Model file not found. Please train and save your model first.")
        return None

model = load_model()
if model is None:
    st.stop()

# === Utility Functions ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_html(html):
    """Extract readable text from HTML safely."""
    if not isinstance(html, str) or not html.strip():
        return ""  # handle NaN or empty
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.extract()
    text = ' '.join([p.get_text() for p in soup.find_all(['p', 'article', 'div'])])
    return clean_text(text)

def get_readability(text):
    try:
        return textstat.flesch_reading_ease(text)
    except:
        return 0.0

def extract_features_from_text(text):
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    flesch_reading_ease = get_readability(text)
    return pd.DataFrame([{
        "word_count": word_count,
        "sentence_count": sentence_count,
        "flesch_reading_ease": flesch_reading_ease
    }])

# === Content Insight Helpers ===
def get_keyword_insights(text, n=10):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [w for w in words if w not in stopwords.words('english')]
    common = Counter(words).most_common(n)
    return pd.DataFrame(common, columns=["Keyword", "Count"])

def detect_content_type(text):
    text = text.lower()
    if re.search(r'\b(price|buy|product|cart)\b', text):
        return "Product Page"
    elif re.search(r'\b(how to|tips|guide|best)\b', text):
        return " Blog Post"
    elif re.search(r'\b(contact|about|service)\b', text):
        return "Company Page"
    elif re.search(r'\b(faq|question|answer)\b', text):
        return " FAQ Page"
    else:
        return " Generic Page"

def estimate_engagement(text):
    sentences = re.split(r'[.!?]', text)
    avg_len = np.mean([len(s.split()) for s in sentences if s.strip()])
    if avg_len > 25:
        return " Long sentences — readability might suffer."
    elif avg_len < 10:
        return "Very short sentences — might feel robotic."
    else:
        return "Balanced sentence length — likely engaging."

def content_depth_insight(word_count):
    if word_count < 300:
        return "Thin content — likely low quality."
    elif word_count < 1000:
        return " Standard length — acceptable."
    else:
        return "In-depth content — likely high quality."

# === Main Analysis ===
def analyze_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SEOQualityBot/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f" Failed to fetch page (status code: {response.status_code})"
    except Exception as e:
        return None, f" Error fetching URL: {e}"

    html = response.text
    text = extract_text_from_html(html)
    if len(text.split()) < 50:
        return None, "Page content too short for reliable analysis."

    features = extract_features_from_text(text)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    result = {
        "url": url,
        "word_count": int(features['word_count'].iloc[0]),
        "sentence_count": int(features['sentence_count'].iloc[0]),
        "readability": round(features['flesch_reading_ease'].iloc[0], 2),
        "quality_label": prediction,
        "probabilities": dict(zip(model.classes_, np.round(proba, 3))),
        "is_thin": len(text.split()) < 300,
        "raw_text": text
    }
    return result, None

def detect_duplicates(texts, urls, threshold=0.8):
    if len(texts) < 2:
        return pd.DataFrame(columns=["url1", "url2", "similarity"])
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    duplicates = []
    for i, j in itertools.combinations(range(len(texts)), 2):
        if sim_matrix[i, j] > threshold:
            duplicates.append({
                "url1": urls[i],
                "url2": urls[j],
                "similarity": float(sim_matrix[i, j])
            })
    return pd.DataFrame(duplicates)

# === Streamlit UI ===
st.set_page_config(page_title="SEO Content Analyzer", layout="wide")
st.title("🔍 SEO Content Quality & Duplicate Detector")

tab1, tab2 = st.tabs(["Single URL", "Batch Mode"])

# --- SINGLE URL MODE ---
with tab1:
    url = st.text_input("Enter a webpage URL:")
    if st.button("Analyze URL") and url:
        with st.spinner("Analyzing..."):
            result, error = analyze_url(url)
            if error:
                st.warning(error)
            elif result:
                st.success(f"✅ Analysis complete for {url}")

                # --- Download buttons (top right) ---
                col_dl1, col_dl2 = st.columns([3, 1])
                with col_dl2:
                    json_data = json.dumps(result, indent=4)
                    st.download_button(
                        label=" Download JSON",
                        data=json_data,
                        file_name="seo_analysis_result.json",
                        mime="application/json",
                        use_container_width=True
                    )

                # --- JSON Summary ---
                st.subheader("📦 Summary Result")
                short_json = {k: v for k, v in result.items() if k != "raw_text"}
                st.json(short_json)

                # --- Content Preview ---
                st.markdown("###  Page Content Preview")
                short_preview = " ".join(result["raw_text"].split()[:50])
                st.write(short_preview + "...")
                with st.expander("Show full content"):
                    st.text(result["raw_text"])

                # --- Metrics ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Word Count", result["word_count"])
                col2.metric("Sentence Count", result["sentence_count"])
                col3.metric("Readability", result["readability"])

                # --- Graphs Side by Side ---
                col_left, col_right = st.columns(2)
                with col_left:
                    proba_df = pd.DataFrame({
                        "Quality": list(result["probabilities"].keys()),
                        "Confidence": list(result["probabilities"].values())
                    })
                    fig = px.bar(
                        proba_df, x="Quality", y="Confidence",
                        color="Quality", text="Confidence",
                        color_discrete_map={"High": "#2ecc71", "Medium": "#f1c40f", "Low": "#e74c3c"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col_right:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["readability"],
                        title={'text': "Readability Score"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3498db"}}
                    ))
                    st.plotly_chart(gauge, use_container_width=True)

                # === 🧠 CONTENT INSIGHTS ===
                st.markdown("##  Content Insights")
                text = result["raw_text"]

                colA, colB = st.columns(2)
                with colA:
                    st.markdown(f"**Content Type:** {detect_content_type(text)}")
                    st.markdown(f"**Depth Analysis:** {content_depth_insight(result['word_count'])}")
                    st.markdown(f"**Engagement:** {estimate_engagement(text)}")

                with colB:
                    st.markdown("**Top Keywords**")
                    keyword_df = get_keyword_insights(text)
                    st.dataframe(keyword_df)

                # === 💡 MAIN INSIGHT SUMMARY ===
                st.info(f"""
**Main Insight:**
This page appears to be a {detect_content_type(text).split()[1]} with **{result['word_count']} words**.
Readability score: **{result['readability']}** ({'good' if result['readability'] > 60 else 'needs improvement'}).
{content_depth_insight(result['word_count'])}
{estimate_engagement(text)}
                """)

                # Separate insights download
                insights_df = pd.DataFrame([{
                    "URL": result["url"],
                    "Content_Type": detect_content_type(text),
                    "Readability": result["readability"],
                    "Word_Count": result["word_count"],
                    "Quality_Label": result["quality_label"],
                    "Engagement": estimate_engagement(text)
                }])
                st.download_button(
                    label=" Download Insights Report (CSV)",
                    data=insights_df.to_csv(index=False),
                    file_name="content_insights.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# --- BATCH MODE ---
with tab2:
    st.write("📂 Upload a CSV with 'url' and 'html_content' columns to detect duplicates and analyze in bulk.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'html_content' not in df.columns:
            st.error("CSV must contain an 'html_content' column.")
        else:
            with st.spinner("Processing..."):
                df = df.dropna(subset=["html_content"]).copy()
                df["html_content"] = df["html_content"].astype(str)
                df["clean_text"] = df["html_content"].apply(extract_text_from_html)
                duplicates_df = detect_duplicates(df["clean_text"], df["url"])
                thin_count = (df["clean_text"].apply(lambda x: len(x.split()) < 500)).sum()

                st.success(f"Found {len(duplicates_df)} duplicate pairs. Thin pages: {thin_count}/{len(df)}")
                st.dataframe(duplicates_df)

                if not duplicates_df.empty:
                    fig_sim = px.histogram(
                        duplicates_df, x="similarity", nbins=20,
                        title="Duplicate Similarity Distribution"
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)

                st.download_button(
                    label=" Download Duplicate Report (CSV)",
                    data=duplicates_df.to_csv(index=False),
                    file_name="duplicates.csv",
                    mime="text/csv",
                    use_container_width=True
                )

st.markdown("---")
st.caption("SEO Content Detector © 2025 | Developed by Data Science Candidate")
