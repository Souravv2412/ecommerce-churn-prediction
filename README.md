# 📊 E-Commerce Customer Churn Analysis & Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-green)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-blue)
![Status](https://img.shields.io/badge/Status-Complete-success)

---

## 🎯 Project Overview

This project analyzes customer churn behavior for an e-commerce platform using **5,630 customer records**. The goal is to identify key churn drivers, build predictive models, and deliver actionable business strategies to reduce churn and increase retention.

**Key Highlights:**
- **Churn Rate:** 16.8% (948 customers)  
- **Best Model:** Decision Tree Classifier (89.3% accuracy, AUC = 0.88)  
- **Critical Finding:** 50% churn in first 3 months  
- **Business Impact:** $236K revenue at risk → $800K retention potential  

---

## 🚀 Live Dashboard

[![View Interactive Dashboard](https://img.shields.io/badge/🔥-View%20Live%20Tableau%20Dashboard-orange?style=for-the-badge)](https://public.tableau.com/views/E-CommerceChurnAnalysisRetentionStrategy/ExecutiveOverview?:showVizHome=no&:embed=true)

**[👉 Click here for full-screen dashboard](https://public.tableau.com/views/E-CommerceChurnAnalysisRetentionStrategy/ExecutiveOverview?:showVizHome=no&:embed=true)**

---

## 📁 Project Structure

```text
E-Commerce Churn Prediction/
├── Business Recommendation/              # Strategic insights & roadmaps
│   ├── 90-Days Implementation Roadmap.jpg
│   ├── Marketing Strategy.jpg
│   └── Retention Recommendations.jpg
├── data/                                  # Dataset files
│   ├── churn_predictions.csv
│   ├── churn_predictions.hyper            # Tableau extract
│   ├── E-Commerce Churn Data.csv          # Raw dataset (5,630 customers)
│   ├── E-Commerce_Cleaned.csv             # Preprocessed data
│   ├── E-Commerce_Cleaned.hyper           # Tableau extract
│   ├── feature_importance.csv
│   ├── feature_importance.hyper           # Tableau extract
│   ├── roc_curve.csv
│   └── roc_curve.hyper                    # Tableau extract
├── images/                                # Key visualizations (19 images)
│   ├── 24f41d6d-e23f-4319-8f7f-440f6a9f3698.jpg
│   ├── cancel.png
│   ├── Cashback Reduces Churn by 50%.png
│   ├── Churn Distribution.png
│   ├── Churn Rate by Tenure.png
│   ├── churn rate.png
│   ├── Complaint impact.png
│   ├── download.png
│   ├── eccomerce logo.jpg
│   ├── eccomerce.png
│   ├── Ecomerce tongle.png
│   ├── eCommerce-logo.png
│   ├── Feature Important.png
│   ├── Filter.png
│   ├── high risk customer.png
│   ├── model Precision.png
│   ├── model tongle.png
│   ├── revenue Loss.png
│   └── ROG Curve.png
├── notebook/                              # Analysis notebooks
│   └── [Jupyter notebooks]
├── ppt/                                   # Presentation files
│   └── [PowerPoint presentations]
├── tableau/                               # Tableau workbooks
│   └── [Tableau files]
├── README.md
└── requirements.txt
```

---

## 🔍 Key Business Insights

### 1️⃣ The New Customer Crisis (Critical Finding)

![Tenure Churn Analysis](images/Churn%20Rate%20by%20Tenure.png)

- **0-3 months:** ~50% churn rate
- **4-6 months:** Drops to ~7.5%
- **After 12 months:** Stabilizes at ~5%

💡 **Business Action:** First 90 days are make-or-break for retention

---

### 2️⃣ Top Churn Drivers (Model Insights)

![Feature Importance](images/Feature%20Importants.png)

| Rank | Feature | Importance | Business Meaning |
|------|---------|------------|------------------|
| 1 | Tenure | 52% | New customers are highest risk |
| 2 | Complaint | 14% | Complaints = 3x churn risk (31.7% vs 10.9%) |
| 3 | NumberOfAddress | 9% | Changing behavior signal |
| 4 | DaySinceLastOrder | 7% | Disengagement warning |
| 5 | CashbackAmount | 4% | Retention tool effectiveness |

---

### 3️⃣ High-Risk Customer Segments

| Segment | Profile | Churn Risk | Strategy |
|---------|---------|------------|----------|
| Frustrated Newcomer | 0-3 months + Complaint | ~50% | 24h escalation SLA |
| Quietly Disengaged | Low cashback + High inactivity | ~25% | Re-engagement campaign |
| Confidently Dissatisfied | Satisfaction 3-4/5 + Silent | ~20% | Deep-dive surveys |

---

## 📊 Strategic Recommendations

![Retention Recommendations](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/Business%20Recomendation/Marketing%20Strategy.jpg)

### 🎯 Four-Pillar Retention Strategy

| Pillar | Target | Action | Impact |
|--------|--------|--------|--------|
| Complaint Response | 31% churn group | <24h escalation | Save $300K |
| Onboarding Focus | 653 new customers | 90-day program | Save $500K |
| Cashback Strategy | High-risk, low engagement | Dynamic rewards | 2x retention boost |
| Satisfaction Fix | 3-4 score customers | Deep-dive surveys | Uncover hidden churn |

---

### 🗓️ 90-Day Implementation Roadmap

![90-Day Roadmap](https://github.com/Souravv2412/ecommerce-churn-prediction/blob/main/Business%20Recomendation/90-Days%20Implementation%20Roadmap.jpg)

- **Phase 1 (Month 1):** Setup → Deploy complaint system, train team
- **Phase 2 (Month 2):** Launch → Onboarding program, early warnings
- **Phase 3 (Month 3):** Optimize → Dynamic cashback, A/B tests
- **Phase 4 (Month 4):** Analyze → Satisfaction paradox investigation
- **Phase 5 (Month 5):** Refine → Scale successful tactics
- **Phase 6 (Month 6):** Measure → Calculate churn reduction & ROI

**Target Outcome:** 16.8% → <12% churn rate = $800K annual retention

---

## 🤖 Machine Learning Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **Decision Tree ⭐** | **89.3%** | **78%** | **52%** | **62%** | **0.88** |
| Logistic Regression | 88.9% | 74% | 52% | 61% | 0.86 |

**Why Decision Tree Won:**
- ✅ Best accuracy (89.3%)
- ✅ Interpretable feature importance
- ✅ Strong AUC (0.88) = reliable predictions
- ✅ Identifies 52% of future churners in advance

![ROC Curve](images/ROG%20Curve.png)

---

## 💰 Business Impact & ROI

### Financial Analysis

| Metric | Value |
|--------|-------|
| Revenue at Risk | $236K annually |
| Customers at Risk | 948 (16.8%) |
| Detectable Churners | ~493 (52% of 948) |
| Potential Savings | ~270 customers (with 55% intervention success) |
| Retention Value | $800K annually |
| Target Churn Rate | <12% (from 16.8%) |

### ROI Calculation

```
Investment:     $50K (tools, campaigns, team)
Return:         $800K (retained revenue)
Net Gain:       $750K
ROI:            1,500%
Payback Period: 1 month
```

---

## 🛠️ Technical Implementation

```python
# Core Pipeline
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Load & Clean
data = pd.read_csv("data/E-Commerce Churn Data.csv")
data.fillna(data.median(numeric_only=True), inplace=True)  # Preserve outliers

# 2. Feature Engineering
X = pd.get_dummies(data.drop(['Churn', 'CustomerID'], axis=1), drop_first=True)
y = data['Churn']

# 3. Train Best Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)  # 89.3% accuracy, AUC 0.88

# 4. Predict & Deploy
predictions = model.predict_proba(X_test)[:, 1]  # Churn probability scores
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 📈 Success Metrics (KPIs)

| KPI | Baseline | Target | Tracking |
|-----|----------|--------|----------|
| Overall Churn Rate | 16.8% | <12% | Monthly |
| New Customer Churn | ~50% | <30% | Weekly |
| Complaint Resolution | 31.7% churn | <20% | Real-time |
| Model Accuracy | 89.3% | >85% | Quarterly |
| Revenue at Risk | $236K | <$150K | Monthly |

---

## 🎓 Key Learnings

1. **First 90 days are everything** → 50% churn happens here
2. **Complaints are opportunities** → 3x risk but preventable with fast response
3. **Satisfaction scores lie** → 3-4/5 scores hide churn risk
4. **Cashback works** → When targeted to high-risk customers
5. **ML enables proactive retention** → 52% detection rate vs. reactive approach

---

## 🔮 Future Enhancements

- [ ] Deploy XGBoost for higher accuracy
- [ ] Real-time API for churn scoring
- [ ] A/B test retention campaigns
- [ ] Customer lifetime value (CLV) prediction
- [ ] Automated email triggers based on risk scores

---

## 👨‍💻 Author

**Souravdeep Singh**  
Data Analyst | Business Intelligence Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/sourav2312/)
[![Email](https://img.shields.io/badge/Email-Contact-red)](mailto:singh.s.deep800@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://souravv2412.github.io/Souravv2412-Souravdeep-Portfolio-Website/)

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- Real-world e-commerce dataset
- Industry best practices in customer analytics
- Feedback from mentors and peers

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📊 Live Dashboard | [View Tableau Public](https://public.tableau.com/views/E-CommerceChurnAnalysisRetentionStrategy/ExecutiveOverview) |
| 📓 Full Analysis | [Jupyter Notebook](notebook/) |
| 📈 Business Strategy | [Recommendations](Business%20Recommendation/) |

---

⭐ **If you found this project helpful, please give it a star!**
