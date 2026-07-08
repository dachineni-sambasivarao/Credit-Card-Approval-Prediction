# Credit Card Approval Prediction Using Machine Learning

## 📌 Project Overview

Banks and financial institutions receive thousands of credit card applications every day. Manual evaluation of these applications is time-consuming and prone to human error. This project automates the credit card approval process using Machine Learning by analyzing an applicant's financial and demographic information to predict whether a credit card application is likely to be approved or rejected.

The project compares multiple machine learning algorithms, selects the best-performing model, and deploys it through a Flask web application for real-time predictions.

---

## 🎯 Objectives

- Automate the credit card approval process.
- Reduce manual effort and decision-making time.
- Improve consistency in credit approval decisions.
- Compare multiple machine learning algorithms.
- Deploy the trained model using a Flask web application.

---

## 📂 Dataset

This project uses two datasets:

- **application_record.csv** – Contains applicant demographic and financial information.
- **credit_record.csv** – Contains historical credit repayment records.

### Dataset Source

Kaggle: https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

### Web Framework

- Flask

### Development Environment

- Google Colab
- Visual Studio Code / PyCharm

---

## 🤖 Machine Learning Algorithms

The following classification algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The best-performing model was selected based on evaluation metrics and saved for deployment.

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 🚀 Project Workflow

1. Environment Setup
2. Data Collection
3. Exploratory Data Analysis (EDA)
4. Data Visualization
5. Data Preprocessing
6. Feature Engineering
7. Model Training
8. Model Evaluation
9. Model Comparison
10. Model Saving
11. Flask Web Application Development
12. Model Deployment

---

## 📁 Project Structure

```
Credit-Card-Approval-Prediction/
│
├── Dataset/
│   ├── application_record.csv
│   └── credit_record.csv
│
├── models/
│   └── best_credit_card_model.pkl
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── requirements.txt
├── README.md
└── Credit_Card_Approval_Prediction.ipynb
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Approval-Prediction.git
```

### Navigate to the Project Folder

```bash
cd Credit-Card-Approval-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Flask application:

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

---

## 🖥️ Web Application Features

- User-friendly input form
- Instant prediction
- Machine Learning-based approval decision
- Responsive interface
- Real-time results

---

## 📈 Model Prediction

The application predicts one of the following outcomes:

- ✅ Credit Card Approved
- ❌ Credit Card Rejected

---

## 💡 Future Enhancements

- Use dropdown menus instead of encoded numeric values.
- Integrate automatic categorical encoding.
- Deploy on IBM Cloud or Render.
- Improve model accuracy through hyperparameter tuning.
- Add user authentication and prediction history.

---

## 👨‍💻 Author

**Shiva Dachineni**

---

## 📜 License

This project is developed for educational and learning purposes.