# 🏦 Credit Risk Prediction App

The **Credit Risk Prediction App** is a machine learning-powered web application that predicts whether a loan applicant is likely to **default** or **repay** a loan. It helps financial institutions and lenders make data-driven decisions by analyzing applicant details such as income, employment, loan amount, credit history, and more.

---

## 🚀 Features

- 📊 **Accurate Predictions:** Predicts loan default risk using trained ML models.  
- 🧠 **Data Preprocessing:** Automatically encodes categorical data and scales numerical data.  
- 💻 **Interactive Interface:** Easy-to-use Streamlit web app for real-time predictions.  
- ⚙️ **Modular Design:** Separate files for model training, saving, and deployment.  
- 📈 **Expandable:** Can be improved with more features or advanced models (e.g., XGBoost, RandomForest).

---

## 🧩 Tech Stack

- **Python 3.10+**  
- **Streamlit** – Frontend web framework  
- **Scikit-learn** – Machine learning model  
- **Pandas & NumPy** – Data preprocessing and analysis  
- **Pickle/Joblib** – Model serialization  
- **Matplotlib/Seaborn** – Data visualization (optional)

---

## 🧠 Model Workflow

1. **Data Collection:** Dataset containing loan applicant information.  
2. **Preprocessing:** Handling missing data, encoding categorical values, and scaling numeric values.  
3. **Model Training:** Training classifier models such as Logistic Regression or Random Forest.  
4. **Model Evaluation:** Using accuracy, precision, recall, F1-score, and ROC-AUC.  
5. **Model Saving:** Exporting trained model and pipeline using `pickle`.  
6. **Deployment:** Streamlit app for user input and real-time prediction.

---
