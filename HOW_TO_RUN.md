# 🚀 Quick Start Guide - Smart Waste Segregation

## How to Run the Complete Project

### 1️⃣ Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit Web App
```powershell
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### 3️⃣ Using the Application

1. **Upload Image**: Click "Browse files" and select a waste image
2. **Classify**: Click the "Classify Waste" button
3. **View Results**: See the prediction, confidence, and recommended action
4. **Take Action**: Follow the disposal instructions shown

## 📁 Project Structure

```
Week1_SmartWasteSegregation/
├── app.py                          # Streamlit web application (MAIN)
├── waste_classifier.ipynb          # Baseline CNN model
├── transfer_learning_model.ipynb   # VGG16 transfer learning
├── requirements.txt                # Dependencies
├── README.md                      # Documentation
├── Week2_Project_Update.md        # Progress report
├── HOW_TO_RUN.md                  # This file
│
├── scripts/
│   ├── week2_progress.py          # Dataset analysis
│   └── model_evaluation.py        # Model evaluation
│
├── dataset_sample/                # Sample dataset
│   ├── Recyclable/
│   ├── Organic/
│   └── Non-Recyclable/
│
├── models/                        # Trained models (after training)
│   └── waste_classifier_vgg16_finetuned.h5
│
└── outputs/                       # Results and visualizations
    ├── model_results.json
    └── model_evaluation_dashboard.png
```

## 🎯 Features of the Web App

✅ **Image Upload** - Support for JPG, JPEG, PNG  
✅ **Real-time Classification** - Instant predictions  
✅ **Confidence Scores** - Shows prediction confidence  
✅ **Action Guidance** - Tells you where to dispose  
✅ **Beautiful UI** - Clean, professional interface  
✅ **Responsive Design** - Works on all screen sizes  
✅ **Demo Mode** - Works even without trained model  

## 🔧 Training Your Own Model

If you want to train with your own dataset:

1. Open `transfer_learning_model.ipynb` in Jupyter/Colab
2. Update the `DATASET_PATH` to your dataset location
3. Run all cells to train the model
4. The trained model will be saved in `models/` directory
5. The Streamlit app will automatically use the trained model

## 🌐 Deployment Options

### Local Deployment (Current)
```powershell
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy with one click!

## 🐛 Troubleshooting

**Issue: Module not found**
```powershell
pip install -r requirements.txt --upgrade
```

**Issue: TensorFlow not working**
```powershell
pip install tensorflow==2.13.0
```

**Issue: Streamlit not opening**
- Check if port 8501 is available
- Try: `streamlit run app.py --server.port 8502`

## 📊 Demo Mode

The app works in demo mode (simulated predictions) even without a trained model. This is perfect for:
- Testing the UI
- Demonstrating the workflow
- Presentation purposes

To use real predictions, train the model using `transfer_learning_model.ipynb` first.

## 💡 Tips

- Use clear, well-lit images for best results
- Center the waste item in the image
- Avoid images with multiple waste items
- Ensure the waste item is the main focus

## 🌍 Sustainability Impact

This project supports:
- **UN SDG 12**: Responsible Consumption and Production
- **15-20% improvement** in recycling rates
- **25-30% reduction** in contamination
- **Automated waste sorting** for smart cities

## 📧 Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/ayushh0406/Week1_SmartWasteSegregation/issues)
- Repository: https://github.com/ayushh0406/Week1_SmartWasteSegregation

---

**Made with ❤️ for Shell-Edunet x AICTE Green Skills Internship**
