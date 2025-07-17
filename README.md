# 🧠 Customer Churn Prediction

## 📌 Project Overview  
This project predicts whether a customer will **churn** (leave the company) based on their behavior and subscription data.  
It uses machine learning and deep learning techniques to help businesses take proactive actions and retain valuable customers.

---

## 📂 Dataset Description  
The dataset contains customer behavior and demographic features such as:

- **Numeric Features:**  
  `Support Calls`, `Payment Delay`, `Total Spend`, `Last Interaction`, `Age`, `Tenure`, `Usage Frequency`

- **Categorical Features:**  
  `Subscription Type` (One-hot encoded)  
  `Contract Length`, `Gender` (Label encoded)

- **Target:**  
  `Churn` (1 = Will leave, 0 = Will stay)

---

## ⚙️ Workflow

1. **Data Cleaning & Preprocessing**
   - Missing values, duplicates, and outlier handling
   - Encoding categorical variables
   - Feature scaling using `StandardScaler`

2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmaps
   - Feature-target relationships
   - Churn distribution

3. **Model Building**
   - Built a deep learning model using **Keras**
   - Evaluation metrics: Accuracy, Confusion Matrix, ROC AUC

4. **Model Saving**
   - Saved model as `model.pkl`
   - Saved scaler as `scaler.pkl` for consistent transformation

---

## 🧪 Results

- **Model Accuracy:** ~89%
- **Strong performance in identifying churned customers**
- Visualized metrics and decision thresholds

---

## 🚀 Deployment Ready  
The project is deployment-ready:
- Saved model + scaler
- Can be integrated into a **Streamlit app** or REST API

---

## 📁 Files in this Repo

| File | Description |
|------|-------------|
| `churn_notebook.ipynb` | Main notebook with all steps explained |
| `model.pkl` | Saved trained model |
| `scaler.pkl` | Saved scaler used during training |
| `requirements.txt` | Python packages used |
| `README.md` | Project overview |

---

## 🔗 How to Use

```bash
# Clone the repo
git clone https://github.com/YourUsername/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Load model and make predictions
