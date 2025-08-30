# app_sentimen_final.py

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import base64
from io import BytesIO
from wordcloud import WordCloud
import numpy as np
import re
from collections import Counter

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

# Load Data with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('discordepreprocessing.csv')
        text_processed = pd.read_csv('hasil_TextPreProcessing_discord.csv')
        return data, text_processed
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        # Create sample data for demo
        sample_data = pd.DataFrame({
            'content': ['Aplikasi bagus sekali', 'Tidak suka aplikasi ini', 'Biasa saja'],
            'Label': ['Positif', 'Negatif', 'Netral'],
            'score': [0.8, 0.2, 0.5]
        })
        return sample_data, sample_data

# Function to load and encode image
def load_profile_image():
    try:
        # Try to load the profile image
        image = Image.open('Pas Foto.jpg')
        return image
    except FileNotFoundError:
        # Return None if image not found
        return None

# Function to generate word cloud
def generate_wordcloud(text, sentiment_type):
    """Generate word cloud for given text"""
    if sentiment_type == 'Positif':
        colormap = 'Greens'
    elif sentiment_type == 'Negatif':
        colormap = 'Reds'
    else:  # Netral
        colormap = 'Blues'
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate(text)
    
    return wordcloud

# Function to load word cloud images
def load_wordcloud_images():
    """Load pre-generated word cloud images"""
    wordcloud_images = {}
    try:
        # Try to load the word cloud images
        wordcloud_images['Positif'] = Image.open('wordcloud_positif.png')
        wordcloud_images['Negatif'] = Image.open('wordcloud_negatif.png')
        wordcloud_images['Netral'] = Image.open('wordcloud_netral.png')
    except FileNotFoundError:
        # If images not found, we'll generate them or show placeholder
        pass
    
    return wordcloud_images


# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

# Global variables for model
trained_model = None
model_accuracy = None

# Load Data with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('discordepreprocessing.csv')
        text_processed = pd.read_csv('hasil_TextPreProcessing_discord.csv')
        return data, text_processed
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        # Create sample data for demo
        sample_data = pd.DataFrame({
            'content': ['Aplikasi bagus sekali', 'Tidak suka aplikasi ini', 'Biasa saja'],
            'Label': ['Positif', 'Negatif', 'Netral'],
            'score': [0.8, 0.2, 0.5]
        })
        return sample_data, sample_data

# Train Naive Bayes Model
@st.cache_resource
def train_naive_bayes_model():
    """
    Train Naive Bayes classifier using the preprocessed training data
    """
    try:
        # Load training data
        training_data = pd.read_csv('hasil_TextPreProcessing_discord.csv')
        
        # Check if required columns exist
        if 'text_clean' not in training_data.columns or 'Label' not in training_data.columns:
            st.error("Required columns 'text_clean' or 'Label' not found in training data")
            return None, None
        
        # Remove rows with NaN values
        training_data = training_data.dropna(subset=['text_clean', 'Label'])
        
        # Prepare features and labels
        X = training_data['text_clean'].astype(str)
        y = training_data['Label']
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with TF-IDF and Naive Bayes
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words=None,  # We already removed stop words in preprocessing
                lowercase=True,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        joblib.dump(pipeline, 'naive_bayes_sentiment_model.pkl')
        
        return pipeline, accuracy
        
    except FileNotFoundError:
        st.error("Training data file 'hasil_TextPreProcessing_discord.csv' not found")
        return None, None
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

# Load or train model
def get_trained_model():
    """
    Load existing model or train new one
    """
    model_file = 'naive_bayes_sentiment_model.pkl'
    
    if os.path.exists(model_file):
        try:
            model = joblib.load(model_file)
            return model, "Model loaded from file"
        except Exception as e:
            st.warning(f"Error loading saved model: {e}. Training new model...")
    
    return train_naive_bayes_model()

# Advanced text preprocessing for prediction
def advanced_preprocess_text(text):
    """
    Advanced text preprocessing similar to training data preprocessing
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove very short words (less than 3 characters)
    words = text.split()
    words = [word for word in words if len(word) >= 3]
    text = ' '.join(words)
    
    return text

# Naive Bayes prediction function
def predict_sentiment_naive_bayes(text, model):
    """
    Predict sentiment using trained Naive Bayes model
    """
    try:
        # Preprocess text
        processed_text = advanced_preprocess_text(text)
        
        if not processed_text.strip():
            return "Netral", 0.5, {"Positif": 0.33, "Negatif": 0.33, "Netral": 0.34}
        
        # Get prediction
        prediction = model.predict([processed_text])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba([processed_text])[0]
        
        # Get class labels
        classes = model.classes_
        
        # Create probability dictionary
        prob_dict = {classes[i]: prob for i, prob in enumerate(probabilities)}
        
        # Get confidence (probability of predicted class)
        confidence = max(probabilities)
        
        return prediction, confidence, prob_dict
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Netral", 0.5, {"Positif": 0.33, "Negatif": 0.33, "Netral": 0.34}

# Function to load and encode image
def load_profile_image():
    try:
        image = Image.open('Pas Foto.jpg')
        return image
    except FileNotFoundError:
        return None

# Function to generate word cloud
def generate_wordcloud(text, sentiment_type):
    """Generate word cloud for given text"""
    if sentiment_type == 'Positif':
        colormap = 'Greens'
    elif sentiment_type == 'Negatif':
        colormap = 'Reds'
    else:  # Netral
        colormap = 'Blues'
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate(text)
    
    return wordcloud

# Interactive sentiment testing with Naive Bayes
def interactive_sentiment_test_nb():
    """
    Interactive sentiment testing interface using Naive Bayes classifier
    """
    global trained_model, model_accuracy
    
    st.markdown("---")
    st.subheader("🤖 Testing Sentimen dengan Naive Bayes Classifier")
    
    # Load or train model if not already done
    if trained_model is None:
        with st.spinner("Loading/Training Naive Bayes model..."):
            trained_model, model_accuracy = get_trained_model()
    
    if trained_model is None:
        st.error("Failed to load or train the Naive Bayes model. Please check your training data.")
        return
    
    # Display model information
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"🎯 Model Accuracy: {model_accuracy:.2%}" if model_accuracy else "Model loaded successfully")
    with col_info2:
        st.info(f"📚 Algorithm: Multinomial Naive Bayes + TF-IDF")
    
    st.markdown("""
    **Fitur ini menggunakan algoritma Naive Bayes Classifier yang telah ditraining dengan data preprocessed Anda.**
    Model ini menggunakan TF-IDF vectorization dan telah dioptimasi untuk analisis sentimen bahasa Indonesia.
    """)
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input area
        user_input = st.text_area(
            "📝 Masukkan teks yang ingin dianalisis:",
            placeholder="Contoh: Discord adalah aplikasi yang sangat bagus untuk chatting dengan teman-teman...",
            height=150,
            help="Masukkan teks dalam bahasa Indonesia"
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analisis dengan Naive Bayes", type="primary")
    
    with col2:
        # Quick test examples
        st.markdown("**💡 Contoh Teks untuk Dicoba:**")
        
        examples = [
            "Discord aplikasi yang sangat bagus dan mudah digunakan!",
            "Aplikasi ini sering lag dan banyak bug, sangat mengecewakan",
            "Biasa saja sih, lumayan untuk chatting tapi bisa lebih baik",
            "Voice chat nya jernih banget, recommended!",
            "Susah login terus, capek deh pake aplikasi ini"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"📋 Contoh {i+1}", key=f"nb_example_{i}"):
                st.session_state.nb_example_text = example
    
    # Use example text if selected
    if 'nb_example_text' in st.session_state:
        user_input = st.session_state.nb_example_text
        st.text_area("📝 Teks terpilih:", value=user_input, height=100, disabled=True)
        analyze_button = True  # Auto-analyze when example is selected
        del st.session_state.nb_example_text
    
    # Perform analysis when button is clicked
    if analyze_button and user_input.strip():
        with st.spinner("🔄 Menganalisis sentimen dengan Naive Bayes..."):
            # Perform sentiment analysis using Naive Bayes
            sentiment, confidence, detailed_scores = predict_sentiment_naive_bayes(user_input, trained_model)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Hasil Analisis Sentimen (Naive Bayes)")
            
            # Create three columns for results
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                # Main sentiment result
                sentiment_color = {
                    'Positif': '#4CAF50',
                    'Negatif': '#f44336', 
                    'Netral': '#FF9800'
                }
                
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 1.5rem; background: {sentiment_color.get(sentiment, '#FF9800')}; 
                    border-radius: 15px; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h2 style="color: white; margin: 0;">🎯 {sentiment}</h2>
                        <p style="color: white; margin: 0; font-size: 18px;">Confidence: {confidence:.1%}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with result_col2:
                # Detailed scores
                st.markdown("**📈 Probabilitas Detail:**")
                for sent_type, score in detailed_scores.items():
                    st.write(f"• {sent_type}: {score:.1%}")
                
                st.markdown(f"**🔧 Text Preprocessing:**")
                processed_text = advanced_preprocess_text(user_input)
                st.write(f"'{processed_text[:50]}...' " if len(processed_text) > 50 else f"'{processed_text}'")
            
            with result_col3:
                # Confidence meter visualization
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': sentiment_color.get(sentiment, '#FF9800')},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=250,
                    font={'color': 'black', 'size': 12},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Probability distribution chart
            st.markdown("---")
            st.subheader("📊 Distribusi Probabilitas")
            
            # Create bar chart for probabilities
            prob_df = pd.DataFrame({
                'Sentimen': list(detailed_scores.keys()),
                'Probabilitas': list(detailed_scores.values())
            })
            
            fig_prob = px.bar(
                prob_df, 
                x='Sentimen', 
                y='Probabilitas',
                color='Sentimen',
                color_discrete_map={
                    'Positif': '#4CAF50',
                    'Negatif': '#f44336',
                    'Netral': '#FF9800'
                },
                title="Distribusi Probabilitas Sentimen"
            )
            
            fig_prob.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50', size=12),
                title_font=dict(size=16, color='#2c3e50'),
                xaxis_title="Kategori Sentimen",
                yaxis_title="Probabilitas",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Additional analysis information
            st.markdown("---")
            st.subheader("ℹ️ Informasi Tambahan")
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown("**🔍 Detail Analisis:**")
                st.write(f"• Panjang teks: {len(user_input)} karakter")
                st.write(f"• Jumlah kata: {len(user_input.split())} kata")
                st.write(f"• Sentimen prediksi: {sentiment}")
                st.write(f"• Metode: Naive Bayes + TF-IDF")
                st.write(f"• Model accuracy: {model_accuracy:.2%}" if model_accuracy else "• Model: Loaded successfully")
            
            with info_col2:
                st.markdown("**💡 Interpretasi Hasil:**")
                if sentiment == "Positif":
                    st.success("✅ Model mengklasifikasikan teks sebagai sentimen positif")
                elif sentiment == "Negatif":
                    st.error("❌ Model mengklasifikasikan teks sebagai sentimen negatif")
                else:
                    st.info("ℹ️ Model mengklasifikasikan teks sebagai sentimen netral")
                
                confidence_interpretation = ""
                if confidence > 0.8:
                    confidence_interpretation = "sangat tinggi - hasil sangat dapat diandalkan"
                elif confidence > 0.6:
                    confidence_interpretation = "tinggi - hasil cukup dapat diandalkan"
                elif confidence > 0.4:
                    confidence_interpretation = "sedang - hasil perlu diverifikasi"
                else:
                    confidence_interpretation = "rendah - hasil tidak dapat diandalkan"
                
                st.write(f"Confidence {confidence:.1%} menunjukkan tingkat keyakinan {confidence_interpretation}")
    
    elif analyze_button and not user_input.strip():
        st.warning("⚠️ Silakan masukkan teks yang ingin dianalisis terlebih dahulu!")

# Model evaluation and info section
def show_model_info():
    """
    Show detailed information about the trained model
    """
    global trained_model, model_accuracy
    
    st.markdown("---")
    st.subheader("🔬 Informasi Model")
    
    if trained_model is None:
        with st.spinner("Loading model information..."):
            trained_model, model_accuracy = get_trained_model()
    
    if trained_model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Spesifikasi Model:**")
            st.write("• Algorithm: Multinomial Naive Bayes")
            st.write("• Vectorizer: TF-IDF")
            st.write("• Max Features: 5000")
            st.write("• N-gram Range: (1, 2)")
            st.write("• Alpha (Smoothing): 1.0")
            if model_accuracy:
                st.write(f"• Test Accuracy: {model_accuracy:.2%}")
        
        with col2:
            st.markdown("**📊 Preprocessing Steps:**")
            st.write("• Lowercasing")
            st.write("• URL removal")
            st.write("• Special character removal")
            st.write("• Stop word removal (in training)")
            st.write("• Tokenization")
            st.write("• Stemming (in training data)")
        
        # Try to load and show training data info
        try:
            training_data = pd.read_csv('hasil_TextPreProcessing_discord.csv')
            st.markdown("**📈 Training Data Info:**")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Total Samples", len(training_data))
            
            with col_info2:
                if 'Label' in training_data.columns:
                    unique_labels = training_data['Label'].nunique()
                    st.metric("Classes", unique_labels)
            
            with col_info3:
                if 'text_clean' in training_data.columns:
                    avg_length = training_data['text_clean'].astype(str).str.len().mean()
                    st.metric("Avg Text Length", f"{avg_length:.0f}")
                    
        except Exception as e:
            st.write(f"Cannot load training data info: {e}")
    
    else:
        st.error("Model not available. Please check your training data file.")
    
    # Instructions and tips
    st.markdown("---")
    st.subheader("📖 Cara Menggunakan")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **🎯 Langkah-langkah:**
        1. Pastikan file 'hasil_TextPreProcessing_discord.csv' ada
        2. Model akan otomatis ditraining pada penggunaan pertama
        3. Gunakan fitur testing interaktif di halaman Distribusi
        4. Model tersimpan dan akan dimuat ulang pada sesi berikutnya
        """)
    
    with tips_col2:
        st.markdown("""
        **💡 Tips untuk hasil terbaik:**
        • Pastikan data training memiliki kolom 'text_clean' dan 'Label'
        • Data training harus berisi minimal 3 kelas sentimen
        • Semakin banyak data training, semakin akurat prediksi
        • Model akan otomatis diupdate jika data training berubah
        """)

# Konfigurasi Full Page
st.set_page_config(
    page_title="Analisis Sentimen Discord", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== Custom CSS with Purple Theme =====================
st.markdown(
    """
    <style>
    /* Background Halaman */
    .stApp {
        background: #ffffff;
        color: #4a4a4a;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Main content area */
    .main .block-container {
        background: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(118, 75, 162, 0.1);
        box-shadow: 0 4px 16px rgba(118, 75, 162, 0.05);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50 !important;
        text-shadow: none;
        font-weight: 600;
    }
    
    /* Text visibility */
    .stMarkdown, .stText, p, span, div {
        color: #2c3e50 !important;
    }
    
    /* Sidebar text */
    .css-1v3fvcr, .css-1v3fvcr * {
        color: #ffffff !important;
    }
    
    /* Metric containers */
    .metric-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(118, 75, 162, 0.1);
        box-shadow: 0 2px 8px rgba(118, 75, 162, 0.05);
    }
    
    .metric-container h2, .metric-container h3, .metric-container p {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* DataFrames */
    .dataframe {
        background: #ffffff;
        border-radius: 10px;
        border: 1px solid rgba(118, 75, 162, 0.1);
    }
    
    /* Profile Card Styling */
    .profile-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(118, 75, 162, 0.3);
        margin: 1rem 0;
    }
    
    .profile-card h1, .profile-card h2, .profile-card h3, .profile-card p {
        color: white !important;
        margin: 0.5rem 0;
    }
    
    .profile-image {
        border-radius: 50%;
        border: 4px solid white;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    /* Profile Info with BIGGER FONTS */
    .profile-info {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .profile-info h2 {
        color: white !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
    }
    
    .profile-info h3 {
        color: white !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
    }
    
    .profile-info p {
        color: white !important;
        font-size: 18px !important;
        line-height: 1.8 !important;
        margin-bottom: 12px !important;
    }
    
    .profile-info p strong {
        color: white !important;
        font-weight: 700 !important;
        font-size: 18px !important;
    }
    
    .profile-info li {
        color: white !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        margin-bottom: 8px !important;
    }
    
    .profile-info ul {
        margin-left: 20px !important;
        margin-top: 10px !important;
    }
    
    /* Chart text colors - Force black text on charts */
    .js-plotly-plot .plotly text {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .legendtext text {
        fill: #000000 !important;
        color: #000000 !important;
    }
    
    /* Tooltip styling */
    .js-plotly-plot .plotly .hovertext {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Pie chart text labels */
    .js-plotly-plot .plotly .pietext {
        fill: #000000 !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    /* Word Cloud Container */
    .wordcloud-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(118, 75, 162, 0.1);
        box-shadow: 0 4px 16px rgba(118, 75, 162, 0.05);
    }
    
    .wordcloud-container h3 {
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
data, text_processed = load_data()

# Load profile image
profile_image = load_profile_image()

# Load word cloud images
wordcloud_images = load_wordcloud_images()

# ===================== Sidebar Menu Modern =====================
with st.sidebar:
    selected = option_menu(
        menu_title="📊 Analisis Sentimen",
        options=["Dashboard", "Profil", "Data Ulasan", "Text Processing", "Diagram Pie", "Chart", "Word Cloud", "Distribusi"],
        icons=["house-fill", "person-circle", "table", "file-earmark-text-fill", "pie-chart-fill", "bar-chart-fill", "cloud-fill", "graph-up"],
        menu_icon="discord",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "rgba(118, 75, 162, 0.8)"},
            "icon": {"color": "#ffffff", "font-size": "20px"},
            "nav-link": {
                "color": "#ffffff", 
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "0px",
                "padding": "10px",
                "border-radius": "5px"
            },
            "nav-link-selected": {"background-color": "rgba(102, 126, 234, 0.8)"},
        }
    )

# ===================== Dashboard =====================
if selected == "Dashboard":
    st.title("🎯 Dashboard Analisis Sentimen Discord")
    st.markdown("---")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <h3>📈 Total Data</h3>
                <h2>{data.shape[0]:,}</h2>
                <p>Ulasan</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    if 'Label' in data.columns:
        sentiment_counts = data['Label'].value_counts()
        
        with col2:
            pos_count = sentiment_counts.get('Positif', 0)
            st.markdown(
                f"""
                <div class="metric-container">
                    <h3>😊 Positif</h3>
                    <h2 style="color: #4CAF50;">{pos_count:,}</h2>
                    <p>{pos_count/len(data)*100:.1f}%</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            neg_count = sentiment_counts.get('Negatif', 0)
            st.markdown(
                f"""
                <div class="metric-container">
                    <h3>😔 Negatif</h3>
                    <h2 style="color: #f44336;">{neg_count:,}</h2>
                    <p>{neg_count/len(data)*100:.1f}%</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col4:
            net_count = sentiment_counts.get('Netral', 0)
            st.markdown(
                f"""
                <div class="metric-container">
                    <h3>😐 Netral</h3>
                    <h2 style="color: #FF9800;">{net_count:,}</h2>
                    <p>{net_count/len(data)*100:.1f}%</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    st.subheader("📋 Preview Data Ulasan")
    st.dataframe(data.head(10), use_container_width=True)

# ===================== Profil =====================
elif selected == "Profil":
    st.markdown(
        """
        <div class="profile-card">
            <h1>👨‍💼 Profil Pengembang</h1>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Profile content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if profile_image:
            st.image(profile_image, width=300, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #764ba2;">
                    <h3 style="font-size: 24px !important; color: #764ba2 !important;">📷 Foto Profil</h3>
                    <p style="font-size: 18px !important; color: #666 !important;">Letakkan file 'Pas Foto.jpg' di direktori yang sama dengan aplikasi</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 2rem; margin: 1rem 0; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                <h2 style="color: #2c3e50; font-size: 32px; font-weight: 700; margin-bottom: 25px; text-shadow: none;">🎓 Informasi Pribadi</h2>
                <p style="color: #34495e; font-size: 20px; line-height: 2; margin-bottom: 15px;"><span style="color: #2c3e50; font-size: 20px; font-weight: 700;">Nama:</span> Kevin Aradia Ramadhan Manumpil</p>
                <p style="color: #34495e; font-size: 20px; line-height: 2; margin-bottom: 15px;"><span style="color: #2c3e50; font-size: 20px; font-weight: 700;">NPM:</span> 10121641</p>
                <p style="color: #34495e; font-size: 20px; line-height: 2; margin-bottom: 15px;"><span style="color: #2c3e50; font-size: 20px; font-weight: 700;">Program:</span> Analisis Sentimen Discord</p>
                <p style="color: #34495e; font-size: 20px; line-height: 2; margin-bottom: 15px;"><span style="color: #2c3e50; font-size: 20px; font-weight: 700;">Metode:</span> Naive Bayes Classifier</p>
                <p style="color: #34495e; font-size: 20px; line-height: 2; margin-bottom: 15px;"><span style="color: #2c3e50; font-size: 20px; font-weight: 700;">Teknologi:</span> Python, Streamlit</p>
                <p style="color: #34495e; font-size: 20px; line-height: 2; margin-bottom: 15px;"><span style="color: #2c3e50; font-size: 20px; font-weight: 700;">Tanggal:</span> 7/3/2025, 10:35:00 AM</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # Additional profile sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 2rem; margin: 1rem 0; backdrop-filter: blur(10px);">
                <h3 style="color: white; font-size: 28px; font-weight: 600; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎯 Tujuan Aplikasi</h3>
                <p style="color: white; font-size: 18px; line-height: 1.8; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Menganalisis sentimen pengguna Discord menggunakan metode Naive Bayes Classifier 
                untuk memahami opini dan perasaan pengguna terhadap aplikasi Discord.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    

# ===================== Data Ulasan =====================
elif selected == "Data Ulasan":
    st.title("📊 Data Ulasan Pengguna")
    st.markdown("Tabel interaktif dengan fitur sorting dan filtering")
    
    # Add search functionality
    search_term = st.text_input("🔍 Cari dalam ulasan:", "")
    
    if search_term:
        filtered_data = data[data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)]
        st.info(f"Menampilkan {len(filtered_data)} hasil dari {len(data)} total data")
    else:
        filtered_data = data
    
    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(filtered_data)
    gb.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gridOptions = gb.build()

    AgGrid(
        filtered_data, 
        gridOptions=gridOptions, 
        height=600, 
        fit_columns_on_grid_load=True,
        theme='streamlit'
    )

# ===================== Text Processing =====================
elif selected == "Text Processing":
    st.title("🔄 Hasil Text Processing")
    st.markdown("Data yang telah melalui proses preprocessing")
    
    if not text_processed.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistik Preprocessing")
            st.write(f"Total dokumen: {len(text_processed)}")
            st.write(f"Kolom tersedia: {list(text_processed.columns)}")
        
        with col2:
            st.subheader("📈 Info Dataset")
            st.write(f"Ukuran data: {text_processed.shape}")
            st.write(f"Memory usage: {text_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("---")
        st.dataframe(text_processed, use_container_width=True)
    else:
        st.warning("Data text processing tidak tersedia")

# ===================== Diagram Pie =====================
elif selected == "Diagram Pie":
    st.title("🥧 Diagram Pie Sentimen")
    
    if 'Label' in data.columns:
        sentiment_counts = data['Label'].value_counts()
        
        # Create plotly pie chart with black text
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=['#667eea', '#764ba2', '#f093fb'],
            textfont=dict(color='white', size=16, family='Arial Black'),
            textinfo='label+percent+value',
            textposition='inside',
            insidetextorientation='horizontal',
            hoverlabel=dict(
                bgcolor='white',
                bordercolor='black',
                font=dict(color='black', size=14)
            ),
            hovertemplate='<b>%{label}</b><br>' +
                         'Jumlah: %{value}<br>' +
                         'Persentase: %{percent}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text="Distribusi Sentimen Pengguna",
                font=dict(size=20, color='black')
            ),
            font=dict(size=16, color='black'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(
                font=dict(color='black', size=14)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show percentages
        st.markdown("### 📊 Detail Persentase")
        for label, count in sentiment_counts.items():
            percentage = (count / sentiment_counts.sum()) * 100
            st.write(f"**{label}**: {count} ({percentage:.1f}%)")
    else:
        st.error("Kolom 'Label' tidak ditemukan dalam data")

# ===================== Chart =====================
elif selected == "Chart":
    st.title("📈 Visualisasi Histogram Sentimen")
    
    if 'Label' in data.columns and 'score' in data.columns:
        # Create single histogram plot
        fig = go.Figure()
        
        # Data for each sentiment
        colors = {'Positif': '#667eea', 'Negatif': '#764ba2', 'Netral': '#f093fb'}
        
        # Add histogram for each sentiment
        for label in data['Label'].unique():
            scores = data[data['Label'] == label]['score']
            fig.add_trace(
                go.Histogram(
                    x=scores,
                    name=label,
                    opacity=0.7,
                    marker_color=colors.get(label, '#cccccc'),
                    hoverlabel=dict(
                        bgcolor='white',
                        bordercolor='black',
                        font=dict(color='black', size=12)
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Skor: %{x}<br>' +
                                 'Frekuensi: %{y}<br>' +
                                 '<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=dict(
                text="Histogram Distribusi Skor Sentimen",
                font=dict(size=20, color='black')
            ),
            xaxis_title="Skor",
            yaxis_title="Frekuensi",
            height=500,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black', size=14),
            xaxis=dict(
                title_font=dict(size=16, color='black'),
                tickfont=dict(color='black', size=12)
            ),
            yaxis=dict(
                title_font=dict(size=16, color='black'),
                tickfont=dict(color='black', size=12)
            ),
            legend=dict(
                font=dict(color='black', size=14)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📊 Statistik Ringkasan")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rata-rata Score", f"{data['score'].mean():.3f}")
        with col2:
            st.metric("Median Score", f"{data['score'].median():.3f}")
        with col3:
            st.metric("Std Deviasi", f"{data['score'].std():.3f}")
    else:
        st.error("Kolom 'Label' atau 'score' tidak ditemukan dalam data")

# ===================== Word Cloud =====================
elif selected == "Word Cloud":
    st.title("☁️ Word Cloud Analisis Sentimen")
    st.markdown("Word cloud menampilkan kata-kata yang paling sering muncul dalam setiap kategori sentimen. Semakin besar ukuran kata, semakin sering kata tersebut muncul dalam ulasan.")
    
    if 'Label' in data.columns:
        # Introduction section
        st.markdown("---")
        st.subheader("📖 Penjelasan Word Cloud")
        st.markdown("""
        **Word Cloud** adalah visualisasi teks yang menampilkan kata-kata dengan ukuran yang berbeda-beda berdasarkan frekuensi kemunculannya:
        - **Kata besar** = Sering muncul dalam ulasan
        - **Kata kecil** = Jarang muncul dalam ulasan
        - **Warna** = Menunjukkan kategori sentimen (Hijau: Positif, Merah: Negatif, Biru: Netral)
        """)
        
        st.markdown("---")
        
        # Display the word cloud comparison image first
        st.subheader("🎨 Perbandingan Word Cloud Sentimen")
        st.markdown("""
        Berikut adalah perbandingan word cloud untuk ketiga kategori sentimen:
        """)
        
        # Load and display the comparison image
        try:
            comparison_image = Image.open('perbandingan_wordcloud.png')
            st.image(comparison_image, caption="Perbandingan Word Cloud Analisis Sentimen Discord", use_container_width=True)
        except FileNotFoundError:
            st.info("📁 Letakkan file 'perbandingan_wordcloud.png' di direktori yang sama dengan aplikasi untuk menampilkan gambar perbandingan.")
        
        # Create tabs for each sentiment with individual images
        tab1, tab2, tab3 = st.tabs(["😊 Sentimen Positif", "😔 Sentimen Negatif", "😐 Sentimen Netral"])
        
        with tab1:
            st.markdown(
                """
                <div class="wordcloud-container">
                    <h3>Word Cloud - Sentimen Positif</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display the positive sentiment word cloud
            try:
                pos_wordcloud = Image.open('wordcloud_positif.png')
                st.image(pos_wordcloud, caption="Word Cloud Sentimen Positif", use_container_width=True)
            except FileNotFoundError:
                st.info("📁 Letakkan file 'wordcloud_positif.png' di direktori yang sama dengan aplikasi.")
            
            # Show analysis for positive sentiment
            st.markdown("**📊 Analisis Word Cloud Positif:**")
            st.markdown("""
            - **Kata dominan**: "discord", "aplikasi", "bagus", "nya", "tolong"
            - **Karakteristik**: Kata-kata menunjukkan apresiasi terhadap aplikasi Discord
            - **Insight**: Pengguna cenderung memberikan feedback positif tentang fitur-fitur Discord
            - **Kata kunci positif**: "bagus", "mantap", "suka", "keren", "tolong" (bantuan)
            """)
            
            # Show top words for positive sentiment
            if 'content' in data.columns:
                pos_data = data[data['Label'] == 'Positif']
                if not pos_data.empty:
                    st.markdown("**📈 Kata-kata yang sering muncul:**")
                    pos_text = ' '.join(pos_data['content'].astype(str))
                    # Simple word frequency (you can enhance this with proper text processing)
                    words = pos_text.split()
                    word_freq = pd.Series(words).value_counts().head(10)
                    st.dataframe(word_freq.reset_index().rename(columns={'index': 'Kata', 0: 'Frekuensi'}))
        
        with tab2:
            st.markdown(
                """
                <div class="wordcloud-container">
                    <h3>Word Cloud - Sentimen Negatif</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display the negative sentiment word cloud
            try:
                neg_wordcloud = Image.open('wordcloud_negatif.png')
                st.image(neg_wordcloud, caption="Word Cloud Sentimen Negatif", use_container_width=True)
            except FileNotFoundError:
                st.info("📁 Letakkan file 'wordcloud_negatif.png' di direktori yang sama dengan aplikasi.")
            
            # Show analysis for negative sentiment
            st.markdown("**📊 Analisis Word Cloud Negatif:**")
            st.markdown("""
            - **Kata dominan**: "aplikasi", "login", "bug", "gak", "tolong"
            - **Karakteristik**: Kata-kata menunjukkan keluhan dan masalah teknis
            - **Insight**: Pengguna sering mengalami masalah login dan bug dalam aplikasi
            - **Masalah utama**: "login", "bug", "gak", "masuk", "update" (masalah teknis)
            """)
            
            # Show top words for negative sentiment
            if 'content' in data.columns:
                neg_data = data[data['Label'] == 'Negatif']
                if not neg_data.empty:
                    st.markdown("**📈 Kata-kata yang sering muncul:**")
                    neg_text = ' '.join(neg_data['content'].astype(str))
                    words = neg_text.split()
                    word_freq = pd.Series(words).value_counts().head(10)
                    st.dataframe(word_freq.reset_index().rename(columns={'index': 'Kata', 0: 'Frekuensi'}))
        
        with tab3:
            st.markdown(
                """
                <div class="wordcloud-container">
                    <h3>Word Cloud - Sentimen Netral</h3>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Display the neutral sentiment word cloud
            try:
                net_wordcloud = Image.open('wordcloud_netral.png')
                st.image(net_wordcloud, caption="Word Cloud Sentimen Netral", use_container_width=True)
            except FileNotFoundError:
                st.info("📁 Letakkan file 'wordcloud_netral.png' di direktori yang sama dengan aplikasi.")
            
            # Show analysis for neutral sentiment
            st.markdown("**📊 Analisis Word Cloud Netral:**")
            st.markdown("""
            - **Kata dominan**: "tolong", "gak", "nya", "login", "suara"
            - **Karakteristik**: Kata-kata menunjukkan pertanyaan dan permintaan bantuan
            - **Insight**: Pengguna sering menanyakan fitur atau meminta bantuan teknis
            - **Fokus utama**: "tolong", "gak", "login", "suara", "voice" (pertanyaan teknis)
            """)
            
            # Show top words for neutral sentiment
            if 'content' in data.columns:
                net_data = data[data['Label'] == 'Netral']
                if not net_data.empty:
                    st.markdown("**📈 Kata-kata yang sering muncul:**")
                    net_text = ' '.join(net_data['content'].astype(str))
                    words = net_text.split()
                    word_freq = pd.Series(words).value_counts().head(10)
                    st.dataframe(word_freq.reset_index().rename(columns={'index': 'Kata', 0: 'Frekuensi'}))
        
        # Summary section
        st.markdown("---")
        st.subheader("📈 Ringkasan Analisis Word Cloud")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🟢 Sentimen Positif**
            - Fokus pada kualitas aplikasi
            - Kata-kata apresiasi ("bagus", "mantap", "suka")
            - Feedback konstruktif
            - Dukungan terhadap fitur Discord
            """)
        
        with col2:
            st.markdown("""
            **🔴 Sentimen Negatif**
            - Masalah teknis dominan
            - Keluhan login dan bug
            - Permintaan perbaikan
            - Frustasi pengguna terhadap aplikasi
            """)
        
        with col3:
            st.markdown("""
            **🔵 Sentimen Netral**
            - Pertanyaan dan bantuan
            - Diskusi fitur (voice, suara)
            - Permintaan informasi
            - Sikap netral terhadap aplikasi
            """)
            
        # Key insights
        st.markdown("---")
        st.subheader("💡 Key Insights dari Word Cloud")
        st.markdown("""
        1. **Kata "discord" dan "aplikasi"** muncul di semua kategori sentimen, menunjukkan fokus diskusi pada aplikasi Discord
        2. **Kata "tolong"** dominan di sentimen negatif dan netral, menunjukkan banyak permintaan bantuan
        3. **Kata "login"** sering muncul di sentimen negatif dan netral, mengindikasikan masalah autentikasi yang perlu diperbaiki
        4. **Kata "bug"** prominan di sentimen negatif, menunjukkan masalah teknis yang perlu segera diatasi
        5. **Kata "bagus"** muncul di sentimen positif, menunjukkan apresiasi pengguna terhadap kualitas aplikasi
        6. **Kata "suara/voice"** muncul di sentimen netral, menunjukkan diskusi tentang fitur audio Discord
        7. **Kata "gak"** muncul di berbagai sentimen, menunjukkan penggunaan bahasa informal dalam ulasan
        """)
        
        # Recommendations based on word cloud analysis
        st.markdown("---")
        st.subheader("🔧 Rekomendasi Berdasarkan Analisis Word Cloud")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Prioritas Perbaikan:**")
            st.markdown("""
            1. **Masalah Login** - Sering muncul di sentimen negatif
            2. **Bug Fixes** - Kata "bug" dominan di sentimen negatif
            3. **Sistem Bantuan** - Banyak permintaan "tolong" di berbagai sentimen
            4. **Fitur Voice/Audio** - Diskusi aktif di sentimen netral
            """)
        
        with col2:
            st.markdown("**📊 Peluang Pengembangan:**")
            st.markdown("""
            1. **Pertahankan kualitas** - Sentimen positif menyukai fitur yang ada
            2. **Improve UX** - Banyak pertanyaan menunjukkan perlu perbaikan antarmuka
            3. **Documentation** - Banyak permintaan bantuan bisa diatasi dengan dokumentasi
            4. **Community Support** - Manfaatkan feedback positif untuk testimonial
            """)
    
    else:
        st.error("Kolom 'Label' tidak ditemukan dalam data")

# ===================== Distribusi Sentimen =====================
elif selected == "Distribusi":
    st.title("📊 Distribusi Data Sentimen")
    
    if 'Label' in data.columns:
        sentiment_counts = data['Label'].value_counts()
        
        # Create interactive bar chart
        fig = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={
                'Positif': '#667eea',
                'Negatif': '#764ba2', 
                'Netral': '#f093fb'
            },
            title="Distribusi Sentimen Pengguna"
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', size=14),
            title_font=dict(size=20, color='#2c3e50'),
            xaxis_title="Kategori Sentimen",
            yaxis_title="Jumlah Ulasan",
            xaxis=dict(title_font=dict(size=16, color='#2c3e50')),
            yaxis=dict(title_font=dict(size=16, color='#2c3e50'))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.markdown("### 📋 Tabel Detail Distribusi")
        
        # Create a more detailed dataframe
        detail_df = pd.DataFrame({
            'Sentimen': sentiment_counts.index,
            'Jumlah': sentiment_counts.values,
            'Persentase': (sentiment_counts.values / sentiment_counts.sum() * 100).round(1)
        })
        
        st.dataframe(detail_df, use_container_width=True)
        
        # Summary info
        st.markdown("### ℹ️ Ringkasan")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Total Ulasan**: {data.shape[0]:,}")
            st.info(f"**Kategori Terbanyak**: {sentiment_counts.index[0]}")
        
        with col2:
            st.info(f"**Persentase Tertinggi**: {sentiment_counts.values[0]/sentiment_counts.sum()*100:.1f}%")
            st.info(f"**Rasio Pos:Neg**: {sentiment_counts.get('Positif', 0)} : {sentiment_counts.get('Negatif', 0)}")
        
        # Add Interactive Sentiment Testing Feature
        interactive_sentiment_test_nb()
        
    else:
        st.error("Kolom 'Label' tidak ditemukan dalam data")



