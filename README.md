
# Early Stage Heart Attack Prediction using Daily Behavioural Data

This machine learning project predicts **early likelihood of Heart Attack** based on daily lifestyle / behavioural data such as weight, smoking history, physical activity, diabetes, alcohol consumption and more.

### Dataset Description

Dataset contains behavioural + medical factors:

| Column | Meaning |
|--------|---------|
| HAD_ATTACK | Target Label (1 = had heart attack event, 0 = no attack) |
| HAD_HEARTDIS | Heart disease history |
| HAD_ASTHMA | Asthma history |
| HAD_DIABETES | Diabetes condition |
| KIDNEY_DIS | Kidney disease |
| DIFFWALK | difficulty walking |
| SMOKING100 | smoked 100 cigarettes lifetime |
| DRINKSIN_30D | drinks in last 30 days |
| PHYSICH_STATUS | physical health status |
| MENTALH_STATUS | mental health status |
| HEALTH_INSU | health insurance |
| PHYSIC_ACT | physical activity |
| SEX | biological sex |
| AGE>65 | age > 65 or not |
| HEAVY_DRINKERS | heavy alcohol |
| Adult | age >= 18 |
| WEIGHT_KG | body weight in KG |

### Goal

> Predict early risk and warn users to modify lifestyle to prevent cardiac event.

### Models Used

- Logistic Regression
- Random Forest
- SVM
- Gradient Boosting (optional)
- SMOTE for class imbalance (optional)

### Steps Done

- Data cleaning
- Encoding + feature transformation
- Train/test split
- Model training
- Model evaluation (Recall is priority)

### Evaluation Metrics

- Accuracy
- Precision
- **Recall (Most Important)**
- F1-score

### Why Recall is priority?

Because in medical domain false negative is dangerous  
→ model should not miss patients having heart risk.

### Future Work

- Mobile app integration for daily health logging
- model Explainability (SHAP/LIME)
- deploy on Flask / Streamlit

### Author

> Abhishek  
(Heart attack risk research project)

---

## Instructions to run

```bash
pip install -r requirements.txt
python train.py
