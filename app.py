from flask import Flask, request, render_template, jsonify
import os
from resume_parser import parse_resume
import joblib
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

salary_model = joblib.load("final_salary_prediction_model.joblib")
label_encoders = joblib.load('label_encoders.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return "No file part in the request"

        resume = request.files['resume']
        
        if resume.filename == '':
            return "No selected file"

        if resume:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume.filename)
            resume.save(filepath)
            parsed_data = parse_resume(open(filepath, 'rb'), resume.filename)
            return render_template('result.html', data=parsed_data)
    
    return render_template('upload.html')


@app.route('/salary', methods=['GET', 'POST'])
def salary():
    if request.method == 'POST':
        try:
            # 1. Read form data
            input_data = {
                'experience_level': request.form['experience_level'],
                'employment_type': request.form['employment_type'],
                'company_location': request.form['company_location'],
                'education_required': request.form['education_required'],
                'years_experience': float(request.form['years_experience']),
                'industry': request.form['industry'],
                'benefits_score': float(request.form['benefits_score']),
                'company_size': request.form['company_size'],
                'job_title': request.form['job_title']
            }

            # 2. Encode categorical features
            for feature, encoder in label_encoders.items():
                if feature in input_data:
                    input_data[feature] = encoder.transform([input_data[feature]])[0]

            # 3. Create DataFrame for prediction
            input_df = pd.DataFrame([input_data])

            # 4. Predict
            prediction = salary_model.predict(input_df)[0]
            predicted_salary = round(prediction, 2)

            return render_template('salary.html', prediction=predicted_salary)
        
        except Exception as e:
            return render_template('salary.html', prediction=f"Error: {str(e)}")

    return render_template('salary.html', prediction=None)

     
if __name__ == '__main__':
    app.run(debug=True)
