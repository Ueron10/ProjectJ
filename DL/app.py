import streamlit as st
import numpy as np
import pickle
import re
import os
import nltk
import time
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Page config
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Initialize preprocessing tools
stemmer = PorterStemmer()
std_stopwords = set(stopwords.words('english'))
sentiment_important = {
    'no', 'not', 'nor', 'don', 'don\'t', 'doesn', 'doesn\'t', 'didn', 'didn\'t',
    'hasn', 'hasn\'t', 'haven', 'haven\'t', 'isn', 'isn\'t', 'aren', 'aren\'t',
    'wasn', 'wasn\'t', 'weren', 'weren\'t', 'be', 'been', 'being', 'have', 'has',
    'had', 'does', 'did', 'will', 'would', 'could', 'ought', 'i', 'you', 'he',
    'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'very', 'so', 'too',
    'just', 'more', 'most', 'such', 'only', 'own', 'same', 'and', 'or', 'if', 'then',
    'because', 'as', 'is', 'are'
}
stop_words = std_stopwords - sentiment_important

@st.cache_resource
def load_artifacts():
    model = load_model(os.path.join(models_dir, "model_final.keras"))
    with open(os.path.join(outputs_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)
    # Ensure correct mapping: index 0 = negative, index 1 = positive
    # If label_classes is ['negative', 'positive'], use as is
    # If reversed, we need to handle it in prediction
    return model, tokenizer, label_classes

# Load artifacts
model, tokenizer, label_classes = load_artifacts()

# Verify label mapping (debug info)
# st.sidebar.write("Label mapping:", {i: label_classes[i] for i in range(len(label_classes))})

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_sentiment(text):
    cleaned = preprocess_text(text)
    if not cleaned:
        return None
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0)[0]
    labels = [str(label).lower() for label in label_classes]
    probabilities = {labels[i]: float(probs[i]) for i in range(len(labels))}
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx])
    return {
        'label': pred_label,
        'confidence': confidence,
        'probabilities': probabilities,
        'cleaned': cleaned
    }

def scrape_imdb_selenium(url, max_reviews=None):
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service as ChromeService
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            try:
                driver = webdriver.Chrome(options=chrome_options)
            except:
                return {"error": f"Chrome driver not found. Please install Chrome browser and chromedriver. Details: {str(e)}", "reviews": [], "count": 0}
        
        driver.get(url)
        time.sleep(5)
        
        reviews_data = []
        last_count = 0
        no_change_count = 0
        max_no_change = 3
        
        while True:
            try:
                load_more = driver.find_element(By.XPATH, '//button[contains(text(),"Load More")]')
                driver.execute_script("arguments[0].click();", load_more)
                time.sleep(3)
            except:
                pass
            
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            reviews = soup.find_all('div', class_="ipc-list-card__content")
            
            current_count = len(reviews)
            
            if max_reviews and len(reviews_data) >= max_reviews:
                break
            
            if current_count == last_count:
                no_change_count += 1
                if no_change_count >= max_no_change:
                    break
            else:
                no_change_count = 0
                last_count = current_count
        
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        reviews = soup.find_all('div', class_="ipc-list-card__content")
        
        limit = max_reviews if max_reviews else len(reviews)
        for review in reviews[:limit]:
            title_elem = review.find('h3')
            title = title_elem.text.strip() if title_elem else ""
            
            content_elem = review.find('div', class_="ipc-html-content-inner-div")
            content = content_elem.get_text(strip=True) if content_elem else ""
            
            if content:
                reviews_data.append({
                    "title": title,
                    "content": content
                })
        
        if driver:
            driver.quit()
        
        if not reviews_data:
            return {"error": "No reviews found", "reviews": [], "count": 0}
        
        return {"reviews": reviews_data, "count": len(reviews_data)}
        
    except Exception as e:
        return {"error": str(e), "reviews": [], "count": 0}
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

st.title("Sentiment Analysis")
st.markdown("Analyze movie review sentiment using CNN")

tab1, tab2 = st.tabs(["Text Input", "IMDB Reviews"])

with tab1:
    st.header("Enter Text")
    user_text = st.text_area("Input your review:", height=200, 
                             placeholder="Example: This movie was absolutely fantastic!")
    
    if st.button("Analyze", key="text_btn"):
        if not user_text.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner("Analyzing..."):
                result = predict_sentiment(user_text)
                if result:
                    col1, col2 = st.columns(2)
                    with col1:
                        sentiment = result['label'].upper()
                        color = "green" if sentiment == "POSITIVE" else "red"
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: {color}20; 
                                    border-left: 5px solid {color};">
                            <h2 style="margin: 0; color: {color};">{sentiment}</h2>
                            <p style="margin: 5px 0 0 0; font-size: 18px;">
                                Confidence: <b>{result['confidence']:.2%}</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Probabilities:**")
                        for label, prob in result['probabilities'].items():
                            st.progress(prob, text=f"{label}: {prob:.2%}")
                    
                    with st.expander("View preprocessed text"):
                        st.text(result['cleaned'])

with tab2:
    st.header("IMDB Reviews (Selenium)")
    st.warning("This mode opens Chrome browser to scrape IMDB JavaScript-loaded reviews. Slower but works for IMDB.")
    
    imdb_url = st.text_input("IMDB Reviews URL:", 
                             placeholder="https://www.imdb.com/title/tt16311594/reviews/",
                             key="imdb_input")
    
    scrape_option = st.radio("Scrape mode:", ["Specific count", "All reviews"])
    
    if scrape_option == "Specific count":
        max_reviews = st.number_input("Number of reviews to scrape:", min_value=1, max_value=100, value=5, step=1)
    else:
        max_reviews = None
        st.info("Will scrape all available reviews until no more 'Load More' button.")
    
    if st.button("Scrape IMDB & Analyze", key="imdb_btn"):
        if not imdb_url.strip():
            st.warning("Please enter IMDB URL!")
        elif "imdb.com" not in imdb_url:
            st.warning("Please enter valid IMDB URL!")
        else:
            with st.spinner("Opening Chrome and scraping... (this may take 30-60 seconds or more for all reviews)"):
                result = scrape_imdb_selenium(imdb_url, max_reviews)
                if result.get("error"):
                    st.error(f"Failed: {result['error']}")
                else:
                    reviews = result['reviews']
                    actual_count = result['count']
                    st.success(f"Scraped {actual_count} reviews from IMDB")
                    
                    st.markdown("### Individual Review Predictions")
                    
                    positive_count = 0
                    negative_count = 0
                    
                    for i, review in enumerate(reviews, 1):
                        review_text = f"{review['title']}: {review['content']}"
                        prediction = predict_sentiment(review_text)
                        
                        if prediction:
                            sentiment = prediction['label'].upper()
                            if sentiment == "POSITIVE":
                                positive_count += 1
                                color = "green"
                            else:
                                negative_count += 1
                                color = "red"
                            
                            with st.expander(f"Review {i}: {sentiment} ({prediction['confidence']:.0%})"):
                                st.markdown(f"**Title:** {review['title']}")
                                st.markdown(f"**Sentiment:** <span style='color:{color};font-weight:bold;'>{sentiment}</span> ({prediction['confidence']:.2%})", unsafe_allow_html=True)
                                st.markdown(f"**Content:** {review['content'][:300]}...")
                    
                    st.markdown("---")
                    st.markdown("### Overall Summary")
                    total = positive_count + negative_count
                    if total > 0:
                        pos_pct = (positive_count / total) * 100
                        neg_pct = (negative_count / total) * 100
                        
                        overall = "POSITIVE" if positive_count >= negative_count else "NEGATIVE"
                        overall_color = "green" if overall == "POSITIVE" else "red"
                        
                        st.markdown(f"**Overall Sentiment:** <span style='color:{overall_color};font-size:24px;font-weight:bold;'>{overall}</span>", unsafe_allow_html=True)
                        st.markdown(f"- Positive: {positive_count} ({pos_pct:.1f}%)")
                        st.markdown(f"- Negative: {negative_count} ({neg_pct:.1f}%)")

st.markdown("---")
st.caption("CNN Model | Binary: Positive/Negative")
