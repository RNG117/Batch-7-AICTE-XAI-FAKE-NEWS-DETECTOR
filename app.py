import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="XAI Fake News Detector", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Explainable AI (XAI) Fake News Detector")
st.markdown("Analyze news articles and see **exactly why** the AI made its decision.")

# ---------------- LOAD SAVED ML MODEL ----------------
@st.cache_resource
def load_ml_models():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("fake_news_model.pkl")
    return tfidf, model

# ---------------- LOAD AI SUMMARIZER ----------------
@st.cache_resource
def load_summarizer():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.tie_word_embeddings = False
    return tokenizer, model

# Initialize models
tfidf, fake_model = load_ml_models()
tokenizer, summarizer = load_summarizer()

# Initialize LIME Explainer
explainer = LimeTextExplainer(class_names=fake_model.classes_)

# LIME requires a pipeline function that outputs probabilities
def lime_predictor(texts):
    vectors = tfidf.transform(texts)
    return fake_model.predict_proba(vectors)

# ---------------- WEB SCRAPER ----------------
def scrape_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])

        return article_text.strip()
    except Exception as e:
        return f"ERROR: {e}"

# ---------------- MAIN ANALYSIS FUNCTION ----------------
def run_analysis(news_text):
    st.write("---")

    # ---- 1. Basic Prediction ----
    vector = tfidf.transform([news_text])
    prediction = fake_model.predict(vector)[0]
    probabilities = fake_model.predict_proba(vector)[0]
    confidence = max(probabilities) * 100

    if prediction == "REAL":
        st.success(f"✅ **VERDICT: {prediction}** (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"🚨 **VERDICT: {prediction}** (Confidence: {confidence:.2f}%)")

    st.write("---")
    col1, col2 = st.columns([1.5, 1])

    # ---- 2. LIME XAI Explanation ----
    with col1:
        st.subheader("Brain of the AI: Why did it choose this?")
        st.caption("Words highlighted in green push the model toward REAL. Words in red push it toward FAKE.")

        with st.spinner("Generating XAI visualization..."):
            exp = explainer.explain_instance(news_text, lime_predictor, num_features=10)
            components.html(exp.as_html(), height=350, scrolling=True)

    # ---- 3. AI Summarization ----
    with col2:
        st.subheader("AI Summary")
        with st.spinner("Generating summary..."):
            input_text = "summarize: " + news_text
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = summarizer.generate(**inputs, max_length=80, min_length=25, do_sample=False)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.info(summary)


# ---------------- USER INPUT TABS ----------------
tab1, tab2 = st.tabs(["📝 Paste Text", "🔗 Enter Article URL"])

with tab1:
    manual_text = st.text_area("Paste your news article below:", height=200, key="text_input")
    if st.button("Analyze Text"):
        if not manual_text.strip():
            st.warning("Please enter some text.")
        else:
            run_analysis(manual_text)

with tab2:
    article_url = st.text_input("Paste the URL of the news article:", placeholder="https://www.example.com/news-article")
    if st.button("Scrape & Analyze URL"):
        if not article_url.strip():
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Fetching article from the web..."):
                scraped_text = scrape_article(article_url)

            if scraped_text.startswith("ERROR:"):
                st.error(f"Could not fetch the article. {scraped_text}")
            elif len(scraped_text) < 100:
                st.warning("Successfully reached the site, but couldn't find enough article text. Try pasting the text manually in the other tab.")
                with st.expander("Show what we found"):
                    st.write(scraped_text)
            else:
                st.success("Article successfully scraped!")
                with st.expander("View Scraped Text"):
                    st.write(scraped_text)
                run_analysis(scraped_text)

st.divider()
st.caption("Powered by Python, Scikit-learn, LIME, Hugging Face, BeautifulSoup, and Streamlit.")
