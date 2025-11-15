"""
Simple Waste Classifier - Lightweight Model
Uses scikit-learn for quick training and deployment
"""

import numpy as np
from PIL import Image
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

def extract_features(image_path):
    """Extract simple features from image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((64, 64))
    
    # Convert to array and flatten
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

def train_model():
    """Train a simple classifier on dataset"""
    print("🔄 Training waste classification model...")
    
    # Try main dataset first, fallback to sample
    dataset_path = Path('dataset')
    if not dataset_path.exists() or len(list(dataset_path.rglob('*.*'))) < 10:
        print("⚠️  Main dataset not found. Using dataset_sample/")
        print("   Run 'python download_dataset.py' to create augmented dataset")
        dataset_path = Path('dataset_sample')
    
    if not dataset_path.exists():
        print("❌ No dataset found!")
        print("   Please ensure dataset/ or dataset_sample/ folder exists")
        return None, None, None
    
    X_train = []
    y_train = []
    
    class_names = ['Non-Recyclable', 'Organic', 'Recyclable']
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    # Load training data
    for class_name in class_names:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue
            
        print(f"📂 Loading {class_name} images...")
        for img_file in class_dir.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    features = extract_features(img_file)
                    X_train.append(features)
                    y_train.append(class_mapping[class_name])
                except Exception as e:
                    print(f"⚠️  Skipped {img_file.name}: {e}")
    
    if len(X_train) == 0:
        print("❌ No training data found!")
        return None, None, None
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"✅ Loaded {len(X_train)} images")
    
    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"   Training: {len(X_train_split)} images")
    print(f"   Testing: {len(X_test)} images")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("🎯 Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train_scaled, y_train_split)
    
    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train_split)
    test_acc = clf.score(X_test_scaled, y_test)
    
    print(f"✅ Training accuracy: {train_acc*100:.2f}%")
    print(f"✅ Testing accuracy: {test_acc*100:.2f}%")
    
    if test_acc < 0.5:
        print("⚠️  Low accuracy - Need more diverse training data")
    
    return clf, scaler, class_names

def save_model(clf, scaler, class_names):
    """Save trained model"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'class_names': class_names
    }
    
    model_path = models_dir / 'waste_classifier_simple.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Model saved to: {model_path}")
    return model_path

def load_trained_model():
    """Load the trained model"""
    model_path = Path('models/waste_classifier_simple.pkl')
    
    if not model_path.exists():
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['classifier'], model_data['scaler'], model_data['class_names']

def predict_image(image, clf, scaler, class_names):
    """Predict waste class for an image"""
    # Save temporarily to extract features
    temp_path = Path('temp_image.jpg')
    image.save(temp_path)
    
    # Extract features
    features = extract_features(temp_path)
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = clf.predict(features_scaled)[0]
    probabilities = clf.predict_proba(features_scaled)[0]
    
    # Clean up
    temp_path.unlink()
    
    predicted_class = class_names[prediction]
    confidence = probabilities[prediction]
    
    return predicted_class, confidence, probabilities

if __name__ == "__main__":
    # Train and save model
    clf, scaler, class_names = train_model()
    
    if clf is not None:
        save_model(clf, scaler, class_names)
        print("\n✅ Model training complete!")
        print("🚀 Now run: streamlit run app.py")
    else:
        print("\n❌ Model training failed!")
        print("📁 Make sure dataset_sample/ folder has images in:")
        print("   - dataset_sample/Recyclable/")
        print("   - dataset_sample/Organic/")
        print("   - dataset_sample/Non-Recyclable/")
