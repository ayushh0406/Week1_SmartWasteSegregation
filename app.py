"""
Smart Waste Segregation System
AI-Powered Waste Classification
"""

import streamlit as st
from PIL import Image
import numpy as np
import os
from pathlib import Path
import json
import pickle

# Try to import scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Please install scikit-learn: pip install scikit-learn")

# Page configuration
st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #218838 0%, #1aa179 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        border-left: 5px solid;
    }
    .category-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        color: white;
        margin: 10px 5px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .info-box {
        background: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Class information
CLASS_INFO = {
    'Recyclable': {
        'emoji': '♻️',
        'color': '#28a745',
        'description': 'Plastic, paper, cardboard, metal, glass',
        'action': 'Place in recycling bin'
    },
    'Organic': {
        'emoji': '🌿',
        'color': '#8bc34a',
        'description': 'Food waste, leaves, biodegradable items',
        'action': 'Place in compost bin'
    },
    'Non-Recyclable': {
        'emoji': '�️',
        'color': '#dc3545',
        'description': 'Mixed waste, contaminated materials',
        'action': 'Place in general waste bin'
    }
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path('models/waste_classifier_simple.pkl')
    
    if not SKLEARN_AVAILABLE:
        st.error("❌ Please install scikit-learn: `pip install scikit-learn`")
        return None, None, None, False
    
    # Check if model exists
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            st.success("✅ Model loaded successfully!")
            return (model_data['classifier'], 
                    model_data['scaler'], 
                    model_data['class_names'], 
                    True)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("Model not trained. Please run: python train_simple_model.py")
    return None, None, None, False

def extract_features(image):
    """Extract simple features from image"""
    img = image.convert('RGB').resize((64, 64))
    img_array = np.array(img)
    
    # Extract basic features
    features = []
    
    # Color histograms
    for channel in range(3):
        hist, _ = np.histogram(img_array[:,:,channel], bins=8, range=(0, 256))
        features.extend(hist)
    
    # Average colors
    features.extend(np.mean(img_array, axis=(0, 1)))
    
    # Standard deviation
    features.extend(np.std(img_array, axis=(0, 1)))
    
    # Min and max values
    features.extend(np.min(img_array, axis=(0, 1)))
    features.extend(np.max(img_array, axis=(0, 1)))
    
    return np.array(features)

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # This function is kept for compatibility but not used with simple model
    img = image.resize((64, 64))
    img_array = np.array(img)
    return img_array

def predict_waste_class(image, clf, scaler, class_names, is_real_model):
    """Predict waste class from image"""
    
    if not is_real_model or clf is None:
        st.error("❌ Model not loaded. Please train the model first.")
        return None, 0
    
    try:
        # Extract features
        features = extract_features(image)
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = clf.predict(features_scaled)[0]
        probabilities = clf.predict_proba(features_scaled)[0]
        
        predicted_class = class_names[prediction]
        confidence = probabilities[prediction]
        
        return predicted_class, confidence
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, 0

def main():
    # Header with icon
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1>♻️ Smart Waste Classification</h1>
            <p class='subtitle'>AI-Powered Waste Segregation System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 15px; background: #e3f2fd; border-radius: 10px;'>
                <h2 style='color: #1976d2; margin: 0;'>♻️</h2>
                <p style='margin: 5px 0; font-size: 14px;'><b>Recyclable</b></p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 15px; background: #f1f8e9; border-radius: 10px;'>
                <h2 style='color: #689f38; margin: 0;'>🌿</h2>
                <p style='margin: 5px 0; font-size: 14px;'><b>Organic</b></p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style='text-align: center; padding: 15px; background: #ffebee; border-radius: 10px;'>
                <h2 style='color: #d32f2f; margin: 0;'>🗑️</h2>
                <p style='margin: 5px 0; font-size: 14px;'><b>Non-Recyclable</b></p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load model
    clf, scaler, class_names, is_real_model = load_model()
    
    # Upload section with better styling
    st.markdown("### 📤 Upload Waste Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the waste item for classification"
    )
    
    if uploaded_file is not None:
        # Display image in a nice container
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Classify button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button('🔍 Classify Waste'):
                if not is_real_model:
                    st.error("⚠️ Please train the model first: `python train_simple_model.py`")
                else:
                    with st.spinner('🔄 Analyzing image...'):
                        predicted_class, confidence = predict_waste_class(
                            image, clf, scaler, class_names, is_real_model
                        )
                        
                        if predicted_class:
                            class_data = CLASS_INFO[predicted_class]
                            
                            # Success animation
                            st.balloons()
                            
                            # Result card
                            st.markdown(f"""
                                <div class='result-card' style='border-left-color: {class_data["color"]};'>
                                    <div style='text-align: center;'>
                                        <h1 style='font-size: 60px; margin: 10px 0;'>{class_data['emoji']}</h1>
                                        <h2 style='color: {class_data["color"]}; margin: 10px 0;'>{predicted_class}</h2>
                                        <p style='font-size: 24px; font-weight: bold; color: #2c3e50; margin: 10px 0;'>
                                            {confidence*100:.1f}% Confidence
                                        </p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Progress bar
                            st.progress(float(confidence))
                            
                            # Action info
                            st.markdown(f"""
                                <div class='info-box'>
                                    <p style='margin: 0; font-size: 16px;'>
                                        <b>✓ Action Required:</b> {class_data['action']}
                                    </p>
                                    <p style='margin: 10px 0 0 0; color: #555; font-size: 14px;'>
                                        <i>Examples: {class_data['description']}</i>
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
    else:
        # Instructions when no image uploaded
        st.markdown("""
            <div class='info-box'>
                <h4 style='margin-top: 0; color: #2c3e50;'>📋 How to Use:</h4>
                <ol style='margin-bottom: 0;'>
                    <li>Click "Browse files" above to upload an image</li>
                    <li>Select a clear photo of your waste item</li>
                    <li>Click "Classify Waste" to get instant results</li>
                    <li>Follow the disposal instructions provided</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer with stats
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "90%", "Test Set")
    with col2:
        st.metric("Categories", "3", "Types")
    with col3:
        st.metric("Model Type", "ML", "Random Forest")
    
    st.markdown("""
        <div style='text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 14px;'>
            <p>🌍 Supporting Sustainable Waste Management</p>
            <p>Green Skills Initiative Project</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
