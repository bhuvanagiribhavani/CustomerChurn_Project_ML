# 📊 Customer Churn Prediction Using Machine Learning

A complete, end-to-end machine learning pipeline that predicts customer churn for a telecom company. Built with **scikit-learn**, the project trains and compares two classifiers — **Logistic Regression** and **Random Forest** — and surfaces actionable business insights from model outputs.



## 🎯 Problem Statement

Customer churn (attrition) is one of the most critical metrics for subscription-based businesses. Acquiring a new customer costs **5–7× more** than retaining an existing one.

This project builds a predictive model that identifies customers likely to churn, enabling proactive retention strategies.



## 📂 Dataset

| Detail | Value |
|--------|--------|
| **Name** | Telco Customer Churn |
| **Source** | IBM Sample Datasets |
| **Rows** | 7,043 |
| **Features** | 21 (demographic, account, service info) |
| **Target** | `Churn` — Yes / No |

Key columns include `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, and more.



## ⚙️ ML Approach

Raw CSV → Exploration → Cleaning & Encoding → Scaling → Train/Test Split  
→ Model Training → Evaluation → Comparison → Insights  

### 🔹 Preprocessing
- Dropped ID columns  
- Converted `TotalCharges` to numeric and handled missing values  
- Encoded categorical variables  
- Standardized numerical features  
- Applied stratified 80/20 train-test split  

### 🔹 Modelling
- Logistic Regression (`class_weight='balanced'`)  
- Random Forest (`class_weight='balanced'`)  

### 🔹 Evaluation
- Accuracy  
- ROC-AUC  
- Confusion Matrix  
- Classification Report  

### 🔹 Interpretation
- Feature importance analysis  
- Business-driven recommendations  



## 🤖 Models Used

| Model | Key Hyperparameters |
|-------|---------------------|
| **Logistic Regression** | `max_iter=2000`, `class_weight='balanced'`, `solver='lbfgs'` |
| **Random Forest** | `n_estimators=200`, `class_weight='balanced'` |



## 📈 Results Summary

| Model | Accuracy | ROC-AUC |
|--------|-----------|----------|
| Logistic Regression | 0.7395 | **0.8396** |
| Random Forest | **0.7892** | 0.8226 |

🏆 **Best model by ROC-AUC:** Logistic Regression (0.8396)  
Since the dataset is imbalanced (~27% churn), ROC-AUC is a more reliable metric than raw accuracy.



## 📉 Visual Evaluation

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

### ROC Curve
![ROC Curve](outputs/roc_curve.png)

### Feature Importance
![Feature Importance](outputs/feature_importance.png)



## 💡 Business Insights

1. **Contract type matters most** — Month-to-month customers churn at significantly higher rates.  
2. **Tenure is protective** — Long-tenured customers rarely leave.  
3. **Service bundling reduces churn** — Customers without security or support services churn more.  
4. **High monthly charges increase risk** — Loyalty incentives may reduce attrition.  
5. **Fiber-optic users churn more** — Service quality or pricing strategies should be reviewed.  
6. **Payment method signal** — Electronic-check payers show higher churn probability.  


## ▶️ How to Run

```bash
# Clone repository
git clone https://github.com/<your-username>/CustomerChurn_Project_ML.git
cd CustomerChurn_Project_ML

# (Optional) Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python churn_model.py
```

All plots and a detailed report are saved automatically in the `outputs/` folder.



## 🗂 Folder Structure

```
CustomerChurn_Project_ML/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── model_results.txt
│
├── churn_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠 Tech Stack

| Tool | Purpose |
|------|----------|
| Python 3.x | Core language |
| pandas / NumPy | Data manipulation |
| scikit-learn | ML models & evaluation |
| matplotlib / seaborn | Visualization |



## 👩‍💻 Author

**Bhuvanagiri Bhavani**  
