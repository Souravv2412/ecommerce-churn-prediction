# 📊 E-Commerce Customer Churn Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.9%252B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%2520Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%2520Models-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 🎯 Project Overview
This project analyzes customer churn behavior for an e-commerce platform using **5,630 customer records**. The goal is to identify key churn drivers, build predictive models, and deliver actionable business strategies to reduce churn and increase retention.

**Key Highlights:**

- **Churn Rate:** 16.8% (948 customers)  
- **Best Model:** Decision Tree Classifier (89.3% accuracy, AUC = 0.88)  
- **Key Drivers:** Tenure (<3 months), Complaints, Low Cashback, Disengagement  
- **Business Impact:** High-risk segments identified and retention strategies proposed  

---

## 📁 Project Structure

```text
ecommerce-churn-prediction-analysis/
├── data/
│   └── E-Commerce Churn Data.csv      # Raw dataset (5,630 customers)
├── notebooks/
│   └── churn_analysis.ipynb          # Full EDA and ML Modeling
├── images/                           # Visualizations for README/Report
│   ├── churn_distribution.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   └── ...                           # (Other impact plots)
├── Churn Analysis Lab 1.pdf          # Final summary report
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
└── LICENSE                           # MIT License

```

---

## 🔍 Key Insights & Visualizations

### 📈 Churn Distribution
![Churn Distribution](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/images/Churn%20Distribution.png)
83.2% active vs 16.8% churned customers

### 📊 Top Churn Drivers (Feature Importance)
![Feature Importance](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/images/Feature%20Important.png)

- **Tenure** – Most important: 50+% weight  
- **Complaints** – Strong early warning signal  
- **CashbackAmount** – Retention tool effectiveness  

### 📉 Churn Rate by Tenure – Critical First 3 Months
![Tenure vs Churn](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/images/Churn%20Rate%20by%20Tenure.png)

- 0-3 months: ~50% churn rate  
- After 6 months: Dramatic drop  
- After 12 months: Minimal churn (<5%)

### 💰 Cashback Reduces Churn by 50%
![Cashback Impact](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/images/Cashback%20Reduces%20Churn%20by%2050%25.png)

- Higher cashback tiers show significantly lower churn rates

### 🚨 Complaints = 80% Churn Risk
![Complaints Impact](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/images/Complaint%20impact.png)

- With complaints: ~80% churn rate  
- Without complaints: ~15% churn rate  

### 📊 ROC Curve – Model Performance
![ROC Curve](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/images/ROG%20Curve.png) 

- **AUC = 0.88** (Strong predictive power)

---

## 🤖 Machine Learning Models

| Model               | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | AUC  |
|--------------------|----------|-----------------|----------------|-----------------|------|
| Decision Tree       | 89.3%    | 0.78            | 0.52           | 0.62            | 0.88 |
| Logistic Regression | 88.9%    | 0.74            | 0.52           | 0.61            | 0.86 |

**Key Findings:**

- Decision Tree performed best with 89.3% accuracy  
- Recall of 52% for churners – identifies half of at-risk customers  
- AUC of 0.88 indicates strong discrimination ability  

---

## 🎯 Business Recommendations

**High-Risk Customer Segments:**

1. **The Frustrated Newcomer**  
   - Tenure: 0-3 months + Has complaints  
   - Churn risk: ~80%  
   - **Action:** Implement 24-hour complaint escalation system  

2. **The Quietly Disengaged**  
   - Low cashback + High days since last order  
   - **Action:** Targeted re-engagement campaigns with personalized offers  

3. **The Confidently Dissatisfied**  
   - Satisfaction score: 3-4 (mid-range)  
   - **Action:** Qualitative follow-up surveys to uncover hidden issues  

**Retention Strategy Roadmap:**

- **Phase 1 (0-3 months):** Enhanced onboarding, welcome offers, proactive check-ins  
- **Phase 2 (3-12 months):** Loyalty programs, personalized recommendations  
- **Phase 3 (12+ months):** VIP benefits, exclusive access, referral programs  

**Target Outcome:** Reduce overall churn rate from 16.8% → <12% within 6 months

---

## 🛠️ Technical Implementation

**Data Pipeline:**

- Data Loading: 5,630 customer records with 20 features  
- Missing Value Handling: Median imputation (preserving outliers as business signals)  
- Feature Engineering: One-hot encoding for categorical variables  
- Train-Test Split: 80-20 stratified split (maintaining churn distribution)  
- Model Training: Logistic Regression + Decision Tree with hyperparameter tuning  

**Key Code Snippets:**

```python
# Data Preprocessing
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("E-Commerce Churn Data.csv")
data.fillna(data.median(numeric_only=True), inplace=True)

# Feature Engineering
X = pd.get_dummies(data.drop(['Churn', 'CustomerID'], axis=1), drop_first=True)
y = data['Churn']

# Model Training
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
```

🚀 How to Run This Project
```
# 1. Clone repository
git clone https://github.com/yourusername/churn-analytics.git
cd churn-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
jupyter notebook notebooks/Churn_Analysis_Full.ipynb

```

Dependencies:
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0

```

## 📊 Skills Demonstrated

| Category             | Technologies / Concepts |
|---------------------|------------------------|
| Data Analysis        | EDA, Statistical Analysis, Data Cleaning, Feature Engineering |
| Machine Learning     | Classification, Model Evaluation, Hyperparameter Tuning, Feature Importance |
| Data Visualization   | Matplotlib, Seaborn, Business Storytelling with Data |
| Business Intelligence| KPI Analysis, ROI Calculation, Strategy Development |
| Tools                | Python, Jupyter, Pandas, Scikit-learn, Git |

---

## 📈 Business Impact & ROI Calculation

- Identified 52% of potential churners in advance  
- Targeted retention could save ~493 customers (52% of 948 churners)  
- Assuming $100 CLV → Potential revenue saved = $49,300  
- Cost of retention campaigns: ~$5,000  
- **Net potential gain:** $44,300  

**Success Metrics:**

- Churn rate reduction (16.8% → <12%)  
- Customer retention cost reduction  
- Increased customer lifetime value (CLV)  
- Improved customer satisfaction scores  

---

## 📚 Learnings & Future Enhancements

**Key Learnings:**

- First 90 days are critical for long-term retention  
- Complaints are gold – early warning signals  
- Cashback is effective if targeted  
- Mid-range satisfaction = hidden churn risk  

**Future Work:**

- Advanced Models: XGBoost, Random Forest, Neural Networks  
- Real-time Prediction: Deploy model as API  
- Dashboard: Power BI / Tableau for business users  
- A/B Testing: Test retention strategies  
- Customer Segmentation: Cluster analysis for personalization  

---

## 👨‍💻 Author

**Your Name**  
Data Analyst | Business Intelligence Specialist  
[LinkedIn Profile](https://www.linkedin.com/in/sourav2312/) | singh.s.deep800@gmail.com 
[Portfolio](https://souravv2412.github.io/Souravv2412-Souravdeep-Portfolio-Website/) 

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.  

---

## 🙏 Acknowledgments

- Dataset from a real-world e-commerce platform  
- Inspired by industry best practices in customer analytics  
- Thanks to mentors and peers for feedback  

---

## 🔗 Quick Links

- 📊 [View Full Analysis Notebook](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/notebook/Churn%20Analysis%20Lab%201.ipynb)
- 📄 [Download Project Report](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/Churn%20Analysis%20Lab%201.pdf)
- 🎤 [View Business Presentation](presentation/Churn_Analysis_Presentation.pptx)  

