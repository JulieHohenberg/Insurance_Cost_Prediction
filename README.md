# 🏥 Medical Insurance Premium Prediction

Final project for **BZAN3307 (Machine Learning & Artificial Intelligence)**  

👉 Dataset: [Medical Insurance Premium Prediction — Kaggle](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction/data)  

---

## 📌 Project Overview
Health insurance premiums have risen significantly in recent years, creating financial uncertainty for families.  
This project develops machine learning models to predict **yearly medical insurance premiums** based on customer demographics and health features, using a simulated dataset of ~1,000 records.  

**Dataset features (10 predictors):**  
- Quantitative: Age, Height, Weight, Number of Major Surgeries  
- Binary: Diabetes, Blood Pressure Problems, Any Transplants, Any Chronic Diseases, Known Allergies, Family Cancer History  
- **Target:** `PremiumPrice` (annual insurance cost, in INR ₹)  

---

## 🔬 Methodology
1. **Data Profiling & EDA**
   - No missing values or sparsity issues  
   - Age ~ Uniform (18–66, mean 42)  
   - Height ~ Normal (145–188 cm, mean 168)  
   - Weight ~ Right-skewed (51–132 kg, mean 77)  
   - Surgeries: 0–3, mostly 0–1  
   - Premiums ranged ₹15k–₹40k, mean ₹24,336, slightly right-skewed  

2. **Model Training & Evaluation**
   Several regression models were trained, tuned, and compared:  
   - **Linear Regression**: train R² = 0.65, test R² = 0.61  
   - **Neural Network Regressor**: reduced overfitting, test R² ≈ 0.65–0.69 (still unstable on small data)  
   - **Support Vector Regression (SVR)**: very poor fit, R² near 0  
   - **Random Forest Regressor**: best performance — train R² = 0.89, test R² = 0.87  
     - Fine-tuning (`min_samples_split=10`) reduced overfitting  

3. **Feature Importance (Random Forest)**
   - Age (70%)  
   - Weight (~13%)  
   - Blood Pressure Problems (~11%)  

4. **Outlier & Transformation Tests**
   - Removing outliers (df_trim) did not improve Random Forest performance and risked excluding under/overweight cases.  
   - Transformations of `PremiumPrice`:  
     - **Log**: improved overall R² to 0.88 but still biased toward low premiums  
     - **Box-Cox (λ=0.75)**: balanced bias, overall R² = 0.85, better at predicting higher premiums  
     - **Quantile transform**: highest test R² (0.90), but extremely biased toward low premiums  

---

## 📈 Key Findings
- **Random Forest Regressor** (with Box-Cox transformed PremiumPrice) was the most **ethical** and balanced model, despite not having the absolute highest R².  
- Feature importance shifted after Box-Cox transformation:  
  - **Chronic Diseases** rose in importance (~10%) → valuable for predicting higher costs.  
  - **Weight** dropped in importance (from ~13% → ~4%).  
  - **Blood Pressure Problems** declined to ~1%.  
- Models tended to **overpredict low premiums** and **underpredict high premiums**, highlighting dataset bias.  
- More representative sampling (stratifying premiums into low/medium/high) would improve fairness.  

---

## ⚖️ Ethical Considerations
- Prioritizing the most accurate model (quantile transform, R² = 0.90) would worsen bias against predicting high premiums accurately.  
- The **Box-Cox Random Forest model** (R² = 0.85) was chosen as the most ethical because it improved prediction for high-cost individuals, where accurate forecasting is most critical.  

---

## 🛠️ Tech Stack
- **Python**  
  - Pandas, NumPy (data preprocessing)  
  - Matplotlib, Seaborn (EDA & visualization)  
  - Scikit-learn (Linear Regression, SVR, Random Forest, Neural Network)  
- **Jupyter Notebook** for workflow  

---

## 📚 Acknowledgments
- **BZAN3307 — Machine Learning & Artificial Intelligence** course final project  
- Dataset by [Tejashvi on Kaggle](https://www.kaggle.com/datasets/tejashvi14/medical-insurance-premium-prediction/data) (CC0 license)  
- Report and analysis by Julie Hohenberg  

---
