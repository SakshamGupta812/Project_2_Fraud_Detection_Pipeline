from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load('models/best_fraud_detection_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    if not file.filename.lower().endswith('.csv'):
        return "Only CSV files are supported", 400

    file_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(file_path)

    try:
        df = pd.read_csv(file_path)

        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)

        required_features = list(model.feature_names_in_)

        missing_columns = [
            column for column in required_features
            if column not in df.columns
        ]

        if missing_columns:
            return f"Missing required columns: {missing_columns}", 400

        df = df[required_features]

        if df.isnull().sum().sum() > 0:
            return "Dataset contains missing values", 400

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df['Prediction'] = predictions
        df['Fraud_Probability'] = probabilities

        total_transactions = len(df)
        fraud_count = int((predictions == 1).sum())
        legitimate_count = int((predictions == 0).sum())

        fraud_percentage = (
            fraud_count / total_transactions
        ) * 100

        results = df[
            ['Amount', 'Prediction', 'Fraud_Probability']
        ].head(100).to_dict(orient='records')

        return render_template(
            'results.html',
            total_transactions=total_transactions,
            fraud_count=fraud_count,
            legitimate_count=legitimate_count,
            fraud_percentage=round(fraud_percentage, 2),
            results=results
        )

    except Exception as e:
        return f"Error processing dataset: {str(e)}", 400


if __name__ == '__main__':
    app.run(debug=True)