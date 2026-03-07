from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI, RateLimitError
import os
from dotenv import load_dotenv

app = Flask(__name__)

# ----------------------------
# Load Trained PCOD Model
# ----------------------------
model = joblib.load('model_joblib.pkl')

# Model Features
features = [
    'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'Cycle(R/I)',
    'Cycle length(days)', 'Marraige Status (Yrs)',
    'Pregnant(Y/N)', 'No. of abortions',
    'Weight gain(Y/N)', 'hair growth(Y/N)',
    'Skin darkening (Y/N)', 'Hair loss(Y/N)',
    'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)'
]

# ----------------------------
# PCOD Risk Category Function
# ----------------------------
def get_risk_category(probability):
    if probability <= 20:
        return "No Significant Risk", "You show minimal signs of PCOS.", "#4CAF50"
    elif probability <= 40:
        return "Low Risk", "Be mindful of symptoms.", "#8BC34A"
    elif probability <= 60:
        return "Moderate Risk", "Consult a gynecologist.", "#FFC107"
    elif probability <= 80:
        return "High Risk", "Consult a doctor.", "#FF9800"
    else:
        return "Very High Risk", "Immediate consultation recommended.", "#F44336"


# ----------------------------
# GAD-7 Function
# ----------------------------
def calculate_gad7(form_data):

    gad_score = sum(int(form_data.get(f'gad{i}', 0)) for i in range(1, 8))

    if gad_score <= 4:
        level = "Minimal Anxiety"
    elif gad_score <= 9:
        level = "Mild Anxiety"
    elif gad_score <= 14:
        level = "Moderate Anxiety"
    else:
        level = "Severe Anxiety"

    return gad_score, level


# ----------------------------
# PHQ-9 Function
# ----------------------------
def calculate_phq9(form_data):

    phq_score = sum(int(form_data.get(f'phq{i}', 0)) for i in range(1, 10))

    if phq_score <= 4:
        level = "Minimal Depression"
    elif phq_score <= 9:
        level = "Mild Depression"
    elif phq_score <= 14:
        level = "Moderate Depression"
    elif phq_score <= 19:
        level = "Moderately Severe Depression"
    else:
        level = "Severe Depression"

    return phq_score, level

def intelligent_suggestion(pcod_level, gad_level, phq_level):

    suggestions = []

    # ------------------------
    # PCOD Level-Based Advice
    # ------------------------
    if pcod_level in ["High Risk", "Very High Risk"]:
        suggestions.append("Follow a low-glycemic diet and avoid processed sugar.")
        suggestions.append("Engage in 30–45 minutes of physical activity daily.")
        suggestions.append("Consult a gynecologist for hormonal evaluation.")

    elif pcod_level == "Moderate Risk":
        suggestions.append("Monitor menstrual cycles regularly.")
        suggestions.append("Maintain balanced nutrition and consistent exercise.")

    else:
        suggestions.append("Maintain a healthy lifestyle and periodic health checkups.")

    # ------------------------
    # Anxiety Level-Based Advice
    # ------------------------
    if gad_level in ["Moderate Anxiety", "Severe Anxiety"]:
        suggestions.append("Practice mindfulness meditation and deep breathing exercises.")
        suggestions.append("Limit caffeine and maintain a regular sleep schedule.")

    elif gad_level == "Mild Anxiety":
        suggestions.append("Incorporate relaxation activities like yoga or journaling.")

    # ------------------------
    # Depression Level-Based Advice
    # ------------------------
    if phq_level in ["Moderate Depression", "Moderately Severe Depression", "Severe Depression"]:
        suggestions.append("Seek guidance from a mental health professional.")
        suggestions.append("Maintain social connections and structured daily routine.")

    elif phq_level == "Mild Depression":
        suggestions.append("Engage in light physical activity and enjoyable hobbies.")

    return " ".join(suggestions)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ai_suggestion(pcod_risk, gad_level, phq_level, user_data):

    prompt = f"""
    A female patient has:
    PCOD Risk: {pcod_risk}
    Anxiety Level: {gad_level}
    Depression Level: {phq_level}
    Symptoms: {user_data}

    Provide:
    1. Lifestyle suggestions
    2. Diet advice
    3. Exercise recommendations
    4. Mental health tips
    Keep response short and medically safe.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception:
        return intelligent_suggestion(pcod_risk, gad_level, phq_level)

# ----------------------------
# Home Route
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')


# ----------------------------
# Prediction Route
# ----------------------------
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = pd.DataFrame([data])

    # ------------------------
    # PCOD Data Processing
    # ------------------------
    numeric_features = ['Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 
                        'Cycle length(days)', 'Marraige Status (Yrs)', 
                        'No. of abortions']

    for feature in numeric_features:
        input_data[feature] = pd.to_numeric(input_data[feature])

    binary_features = ['Pregnant(Y/N)', 'Weight gain(Y/N)', 
                       'hair growth(Y/N)', 'Skin darkening (Y/N)', 
                       'Hair loss(Y/N)', 'Pimples(Y/N)', 
                       'Fast food (Y/N)', 'Reg.Exercise(Y/N)']

    for feature in binary_features:
        input_data[feature] = input_data[feature].astype(int)

    input_data['Cycle(R/I)'] = 1 if input_data['Cycle(R/I)'].iloc[0].lower() == 'irregular' else 0

    input_data = input_data[features]

    # ------------------------
    # PCOD Prediction
    # ------------------------
    probability = model.predict_proba(input_data)[0][1] * 100
    category, advice, color = get_risk_category(probability)

    # CREATE RESULT DICTIONARY  ✅ IMPORTANT
    result = {
        'probability': round(probability, 2),
        'category': category,
        'advice': advice,
        'color': color
    }

    # ------------------------
    # GAD-7 Calculation
    # ------------------------
    gad_score = sum(int(data.get(f'gad{i}', 0)) for i in range(1, 8))

    if gad_score <= 4:
        gad_level = "No Anxiety"
    elif gad_score <= 9:
        gad_level = "Mild Anxiety"
    elif gad_score <= 14:
        gad_level = "Moderate Anxiety"
    else:
        gad_level = "Severe Anxiety"

    # ------------------------
    # PHQ-9 Calculation
    # ------------------------
    phq_score = sum(int(data.get(f'phq{i}', 0)) for i in range(1, 10))

    if phq_score <= 4:
        phq_level = "No Depression"
    elif phq_score <= 9:
        phq_level = "Mild Depression"
    elif phq_score <= 14:
        phq_level = "Moderate Depression"
    elif phq_score <= 19:
        phq_level = "Moderately Severe Depression"
    else:
        phq_level = "Severe Depression"

    ai_suggestion = generate_ai_suggestion(
    category,
    gad_level,
    phq_level,
    data
    )
    # ------------------------
    # Return All Results
    # ------------------------
    return render_template(
        'result.html',
        result=result,
        probability=result['probability'],
        category=result['category'],
        gad_score=gad_score,
        gad_level=gad_level,
        phq_score=phq_score,
        phq_level=phq_level,
        ai_suggestion=ai_suggestion
    )


if __name__ == '__main__':
    app.run(debug=True)