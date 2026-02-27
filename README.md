# Explainable AI (XAI) Fake News Detector

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/AI-Scikit--Learn-orange)
![Hugging Face](https://img.shields.io/badge/GenAI-HuggingFace-yellow)

## 📌 Project Overview
The rapid spread of online disinformation is a critical threat to digital security and public trust. While standard machine learning models can detect fake news, they often operate as opaque "black boxes." 

This capstone project is a **hybrid Artificial Intelligence system** designed to combat digital disinformation transparently. It combines traditional Machine Learning for authenticity classification, Explainable AI (LIME) to visually prove the model's reasoning, and Generative AI (Google Flan-T5) for abstractive text summarization.

## 🚀 Key Features
* **Dual Input Interface:** Analyze manually pasted text or scrape live news articles directly via their URL (powered by `BeautifulSoup`).
* **Authenticity Classification:** A Logistic Regression model trained on TF-IDF vectors classifies the news as **REAL** or **FAKE** alongside a mathematical confidence score.
* **Explainable AI (XAI):** Integrates **LIME** (Local Interpretable Model-agnostic Explanations) to visually highlight the exact sensationalized or factual words that triggered the AI's decision.
* **GenAI Summarization:** Uses a pre-trained **Flan-T5 Large Language Model** to condense lengthy, time-consuming articles into quick, digestible summaries.

## 🛠️ Technology Stack
* **Programming Language:** Python
* **Machine Learning:** `scikit-learn`, `pandas`, `joblib`
* **Explainable AI (XAI):** `lime`
* **Natural Language Processing (NLP):** `transformers`, `torch` (Hugging Face)
* **Web Scraping:** `beautifulsoup4`, `requests`
* **Frontend UI:** `streamlit`

## 📂 Repository Structure
* `app.py`: The main Streamlit web application script containing the UI, scraping logic, and model inference pipelines.
* `news_dataset.csv`: The base dataset used for training the initial classification model.
* `fake_news_model.pkl`: The saved, pre-trained Logistic Regression classification model.
* `tfidf_vectorizer.pkl`: The saved TF-IDF text vectorizer.
* `requirements.txt`: List of required Python dependencies for deployment.

## 💻 How to Run Locally

**1. Clone this repository:**
```bash
git clone https://github.com/RNG117/Batch-7-AICTE-XAI-FAKE-NEWS-DETECTOR.git
cd Batch-7-AICTE-XAI-FAKE-NEWS-DETECTOR
```
**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Launch the Streamlit application:**
```bash
streamlit run app.py
```

*Note: The app will automatically open in your default web browser at http://localhost:8501. The Flan-T5 summarization model will be downloaded from Hugging Face automatically upon the first run.*

## Future Scope
* **Multilingual Support:** Expanding NLP capabilities to detect disinformation in regional languages.
* **Domain Reputation Analysis:** Integrating OSINT APIs (like VirusTotal or WHOIS) to cross-reference URL domain age and credibility.
* **Deepfake Text Detection:** Upgrading the classification pipeline to detect AI-generated text payloads.

## Author
**Rujaal Ghate** | B.Tech Computer Science & Engineering (Cybersecurity)  
*Capstone Project*
