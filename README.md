
# 🌿 Smart Waste Segregation using CNN  
*(Image Classification for Recyclable, Organic, and Non-Recyclable Waste)*  

---

## 🔍 **Project Title**  
**AI-Powered Waste Classifier** — Automatically classifies waste into **Recyclable**, **Organic**, and **Non-Recyclable** categories using **Convolutional Neural Networks (CNN)**.

---

## 🧩 **Problem Statement**  
Improper waste segregation is one of the biggest environmental challenges faced by modern cities.  
A large percentage of recyclable materials end up in landfills due to inefficient manual sorting.  
By leveraging AI, we can automate the process of classifying waste images, **improve recycling efficiency**, and **reduce environmental pollution** — contributing to a cleaner and sustainable ecosystem.

---

## 🎯 **Goal**
- Develop a CNN-based model to classify waste images accurately.  
- Achieve high accuracy for the three categories: Recyclable, Organic, and Non-Recyclable.  
- Promote automation in waste management using AI for sustainability.  
- Support **UN Sustainable Development Goal 12 – Responsible Consumption and Production**.

---

## 🛠 **Tools & Technologies**
- **Programming Language:** Python (3.x)  
- **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn  
- **Development Environment:** Google Colab (GPU runtime enabled)  
- **Dataset Source:** [Waste Classification Dataset – Kaggle](https://www.kaggle.com/datasets/phenomsg/waste-classification)  

---

## 📁 **Dataset Overview**
- **Total Images:** ~30,000  
- **Classes:**  
  - ♻️ *Recyclable* (plastic, paper, metal, cardboard, etc.)  
  - 🌿 *Organic* (food waste, leaves, biodegradable items)  
  - 🚯 *Non-Recyclable* (general trash, mixed waste)  
- **Data Split:** 80% training | 20% validation  
- **Pre-processing:** Image resizing (150×150 px), normalization (0–1), and augmentation (rotation, shift, flip)  

---

## ⚙️ **Methodology – Week 1**
1. **Data Collection:** Downloaded and organized the dataset from Kaggle into labeled directories.  
2. **Preprocessing:** Performed image normalization, resizing, and data augmentation to improve model generalization.  
3. **Model Design:**  
   Built a CNN architecture with convolutional and pooling layers followed by dense layers for classification.
```

Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
→ Flatten → Dense(512, relu) → Dropout(0.5) → Dense(3, softmax)

```
4. **Compilation:** Used *Adam* optimizer, *categorical_crossentropy* loss, and accuracy as a metric.  
5. **Training:** Trained the model for 5 epochs as a baseline to validate data quality.  
6. **Evaluation:** Visualized accuracy & loss curves to analyze performance.  
7. **Output:** Saved trained model (`waste_classifier.h5`) and prediction results for sample images.

---

## 📊 **Results – Week 1 Baseline**
| Metric | Value (Example) |
|:-------:|:---------------:|
| **Training Accuracy** | ~88% |
| **Validation Accuracy** | ~82% |
| **Epochs** | 5 |
| **Classes** | 3 (Recyclable, Organic, Non-Recyclable) |



---

## 🌍 **Sustainability Impact**
This project supports **Sustainable Development Goal 12 – Responsible Consumption & Production** by promoting efficient recycling.  
Through AI-powered segregation, we can:  
- Reduce landfill waste by **20–30%**  
- Improve recycling rates significantly  
- Encourage environmentally conscious waste handling practices  

---

## 📂 **Repository Structure**
```

Week1_SmartWasteSegregation/
│
├── waste_classifier.ipynb        ← Jupyter Notebook (main code)
├── README.md                     ← Project details
├── requirements.txt               ← Python dependencies
│
├── dataset_sample/
│   ├── Recyclable/
│   ├── Organic/
│   └── Non-Recyclable/
│
├── output_samples/
│   ├── accuracy_plot.png
│   └── sample_prediction.png
│
└── waste_classifier.h5           ← Saved model (optional)

```

---

## 🔮 **Next Steps (Week 2 & Week 3)**
- Implement **Transfer Learning (VGG16/ResNet50)** to improve accuracy beyond 90%.  
- Tune hyperparameters for optimal model performance.  
- Develop a **Streamlit web app** for real-time waste classification.  
- Prepare final PPT and present results to mentors.

---

## 🗂️ Week 2 Progress — Project Update (≈60% complete)

This repo was updated to reflect Week 2 progress. Key additions and changes made this week:

- `Week2_Project_Update.md` — a short progress report describing the work done this week and the plan to finish the project.
- `scripts/week2_progress.py` — a small, dependency-free script that inspects `dataset_sample/` and prints image counts per class (quick EDA/check).
- `README.md` updated with this Week 2 summary and instructions to run the script.

What "~60% complete" covers:
- Dataset organized and verified in `dataset_sample/`.
- Baseline model and training notebook present (`waste_classifier.ipynb`) with an initial run.
- Basic EDA script added and documented.
- Plan for Transfer Learning, hyperparameter tuning, and deployment (Streamlit) created.

Quick check (PowerShell):

```powershell
# From repository root
python .\scripts\week2_progress.py
```

See `Week2_Project_Update.md` for the detailed weekly log, files added, sample outputs, and next steps to reach final submission.


## 🧠 **References**
- Kaggle Dataset: [Waste Classification Dataset](https://www.kaggle.com/datasets/phenomsg/waste-classification)  
- TensorFlow Documentation: [https://www.tensorflow.org](https://www.tensorflow.org)  
- Keras CNN Tutorial: [https://keras.io/examples/vision/image_classification_from_scratch/](https://keras.io/examples/vision/image_classification_from_scratch/)  

---

