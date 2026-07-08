from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model (full pipeline: preprocessing + Random Forest)
model = joblib.load("models/best_credit_card_model.pkl")

# Raw column order/names the pipeline expects as input
FEATURE_COLUMNS = joblib.load("models/feature_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Collect input data from the HTML form.
        # NOTE: categorical fields (CODE_GENDER, FLAG_OWN_CAR, etc.) must be
        # the ORIGINAL string categories (e.g. 'M'/'F', 'Y'/'N'), because the
        # saved pipeline's OneHotEncoder was fit on those raw string values -
        # not on manually-encoded integers.
        input_dict = {
            'CODE_GENDER': request.form['CODE_GENDER'],
            'FLAG_OWN_CAR': request.form['FLAG_OWN_CAR'],
            'FLAG_OWN_REALTY': request.form['FLAG_OWN_REALTY'],
            'CNT_CHILDREN': int(request.form['CNT_CHILDREN']),
            'AMT_INCOME_TOTAL': float(request.form['AMT_INCOME_TOTAL']),
            'NAME_INCOME_TYPE': request.form['NAME_INCOME_TYPE'],
            'NAME_EDUCATION_TYPE': request.form['NAME_EDUCATION_TYPE'],
            'NAME_FAMILY_STATUS': request.form['NAME_FAMILY_STATUS'],
            'NAME_HOUSING_TYPE': request.form['NAME_HOUSING_TYPE'],
            'DAYS_BIRTH': int(request.form['DAYS_BIRTH']),
            'DAYS_EMPLOYED': int(request.form['DAYS_EMPLOYED']),
            'FLAG_MOBIL': int(request.form['FLAG_MOBIL']),
            'FLAG_WORK_PHONE': int(request.form['FLAG_WORK_PHONE']),
            'FLAG_PHONE': int(request.form['FLAG_PHONE']),
            'FLAG_EMAIL': int(request.form['FLAG_EMAIL']),
            'OCCUPATION_TYPE': request.form['OCCUPATION_TYPE'],
            'CNT_FAM_MEMBERS': float(request.form['CNT_FAM_MEMBERS']),
        }

        # Build DataFrame in the exact column order the pipeline was trained on
        input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)

        # Prediction
        prediction = model.predict(input_df)[0]

        if prediction == 0:
            result = "Credit Card Approved"
        else:
            result = "Credit Card Rejected"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
