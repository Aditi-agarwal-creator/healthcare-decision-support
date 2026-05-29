# 🩺 AI-Powered Disease Risk Prediction System
### JIIT Noida | CSE 4th Sem AI-ML Project
**Team:** Agrima Gupta | Parisha | Aditi Agrawal  
**Guide:** Dr. Mukta Goyal

---

## 📋 PROJECT OVERVIEW
A two-phase AI system that:
- **Phase 1:** Predicts disease risk (Low/Medium/High) from lifestyle data
- **Phase 2:** Predicts specific disease from symptoms (if risk is Medium/High)
- Gives personalized health recommendations

---

## 🚀 STEP-BY-STEP SETUP GUIDE

### ✅ STEP 1 — Install Python
1. Download Python 3.10+ from https://python.org
2. During install → ✅ Check "Add Python to PATH"
3. Open Command Prompt → type: `python --version` (should show version)

---

### ✅ STEP 2 — Create Project Folder
```
Open CMD / Terminal and type:

mkdir disease_prediction
cd disease_prediction
```

---

### ✅ STEP 3 — Install VS Code (Code Editor)
1. Download from https://code.visualstudio.com
2. Install Python extension inside VS Code
3. Open your project folder in VS Code

---

### ✅ STEP 4 — Install Required Libraries
```bash
pip install streamlit scikit-learn numpy pandas matplotlib
```
OR use requirements.txt:
```bash
pip install -r requirements.txt
```

---

### ✅ STEP 5 — Copy Project Files
Make sure these files are in your folder:
```
disease_prediction/
│
├── app.py              ← Main application file
├── requirements.txt    ← All libraries listed
└── README.md           ← This file
```

---

### ✅ STEP 6 — Run The App
```bash
streamlit run app.py
```
Browser will auto-open at: http://localhost:8501

---

## 📂 WHAT EACH FILE DOES

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web app — UI + ML models |
| `requirements.txt` | All Python libraries needed |
| `README.md` | Setup guide (this file) |

---

## 🧠 HOW THE CODE WORKS

### Data Generation (inside app.py)
Since we don't have a fixed CSV dataset, synthetic data is generated using `numpy`:
- **generate_lifestyle_data()** → Creates 2000 patient lifestyle records
- **generate_symptom_data()** → Creates 2000 symptom-disease mappings

### Model Training
4 models are trained automatically when app starts:
1. **Random Forest** — Best performer
2. **Logistic Regression**
3. **Decision Tree**  
4. **k-NN**

### Prediction Flow
```
User fills form → Data preprocessed → Phase 1 model predicts risk
                                         ↓
                              Low → Recommendations only
                              Medium/High → Go to Phase 2
                                              ↓
                                   User enters symptoms
                                              ↓
                                   Phase 2 predicts disease
                                              ↓
                                   Show recommendations
```

---

## 🔧 HOW TO USE REAL DATASET (Optional - Better Accuracy)

Download from Kaggle:
- **Diabetes:** https://kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Heart Disease:** https://kaggle.com/datasets/fedesoriano/heart-failure-prediction

Then replace `generate_lifestyle_data()` with:
```python
df = pd.read_csv("diabetes.csv")
```
And map columns accordingly.

---

## 📊 FEATURES IMPLEMENTED

- [x] Lifestyle-based risk prediction (Phase 1)
- [x] Symptom-based disease prediction (Phase 2)  
- [x] Risk classification: Low / Medium / High
- [x] Confidence scores for each prediction
- [x] 4 ML algorithm comparison
- [x] Model accuracy charts
- [x] Personalized recommendations
- [x] Dark theme professional UI
- [x] Ethical AI disclaimer

---

## 💡 FUTURE IMPROVEMENTS (For Better Marks)

1. **PDF Report Generation** — Patient can download their report
2. **SHAP Explainability** — Show WHY model made that prediction
3. **Real Kaggle Datasets** — Better accuracy
4. **User Login System** — Save patient history
5. **Chatbot Interface** — Talk to AI doctor
6. **Email Alerts** — Send health report to email

---

## 🗂️ HOW TO SUBMIT

1. Zip the entire folder → `disease_prediction.zip`
2. Record a screen demo video (3-5 minutes)
3. Prepare PPT with: Problem → Solution → Architecture → Demo → Results

---

## ⚕️ DISCLAIMER
This system is for educational purposes only.  
It does NOT provide medical diagnosis.  
Always consult a qualified doctor for health decisions.

---
*JIIT Noida | B.Tech CSE | 2025-26*
