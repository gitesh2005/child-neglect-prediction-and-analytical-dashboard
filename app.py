from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import json
import requests
import os
from dotenv import load_dotenv

# Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ----------------------------------------
# Config / Paths
# ----------------------------------------
MODEL_PATH = os.path.join('models', 'child_neglect_xgb_model.pkl')
SCALER_PATH = os.path.join('models', 'child_neglect_scaler.pkl')
FEATURE_COLS_PATH = os.path.join('models', 'feature_cols.json')
DATA_CSV = os.path.join('data', 'dstrCAC.csv')

# ----------------------------------------
# Load ML Model and Preprocessing Objects
# ----------------------------------------
model = None
scaler = None
feature_cols = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, 'r') as f:
            feature_cols = json.load(f)
    print("✓ Model/scaler/feature_cols loaded.")
except Exception as e:
    print(f"✗ Error loading model artifacts: {e}")
    model, scaler, feature_cols = None, None, None

# ----------------------------------------
# Load district CSV
# ----------------------------------------
df = None
try:
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        print("✓ Dataset loaded for district lookup")
    else:
        print("✗ Dataset CSV not found at", DATA_CSV)
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    df = None

# ----------------------------------------
# OpenRouter API Configuration (GenAI)
# ----------------------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ----------------------------------------
# Routes: Serve pages
# ----------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# ----------------------------------------
# Prediction endpoint
# ----------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'ML model or scaler not loaded.'}), 500

        data = request.get_json(force=True, silent=True) or {}
        def to_f(x): 
            try:
                return float(x)
            except:
                return 0.0

        murder = to_f(data.get('murder', 0))
        rape = to_f(data.get('rape', 0))
        kidnapping = to_f(data.get('kidnapping', 0))
        foeticide = to_f(data.get('foeticide', 0))
        abetment = to_f(data.get('abetment', 0))
        procuration = to_f(data.get('procuration', 0))
        buying = to_f(data.get('buying', 0))
        selling = to_f(data.get('selling', 0))
        child_marriage = to_f(data.get('child_marriage', 0))
        other_crimes = to_f(data.get('other_crimes', 0))

        violent_crimes = murder + rape + kidnapping
        sexual_exploitation = buying + selling + procuration
        child_protection = foeticide + child_marriage + abetment
        total = murder + rape + kidnapping + foeticide + abetment + procuration + buying + selling + child_marriage + other_crimes
        crime_ratio = (violent_crimes + 1) / (other_crimes + 1)
        prostitution_to_total = (buying + selling) / (total + 1)
        other_to_total = other_crimes / (total + 1)

        features = [
            murder, rape, kidnapping, foeticide, abetment,
            procuration, buying, selling, child_marriage, other_crimes,
            violent_crimes, sexual_exploitation, child_protection,
            crime_ratio, prostitution_to_total, other_to_total
        ]

        features_arr = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_arr)

        prediction = model.predict(features_scaled)[0]
        try:
            prob = model.predict_proba(features_scaled)[0]
            probability = round(float(prob[1] * 100), 2)
        except Exception:
            probability = None

        result = {
            'prediction': 'High Risk' if int(prediction) == 1 else 'Low Risk',
            'prediction_value': int(prediction),
            'probability': probability
        }

        return jsonify(result)

    except Exception as e:
        print("Error in /predict:", str(e))
        return jsonify({'error': str(e)}), 500

# ----------------------------------------
# GenAI endpoint
# ----------------------------------------
@app.route('/genai', methods=['POST'])
def genai():
    try:
        data = request.get_json(force=True, silent=True) or {}
        user_query = data.get('query', '').strip()

        if not user_query:
            return jsonify({'error': 'No query provided'}), 400

        if not OPENROUTER_API_KEY:
            return jsonify({'error': 'OpenRouter API key not configured on server.'}), 500

        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': 'openai/gpt-3.5-turbo',
            'messages': [
                {'role': 'system', 'content': 'You are an expert in child protection and data analytics. Provide concise, actionable insights based on crime data and ML predictions.'},
                {'role': 'user', 'content': user_query}
            ],
            'max_tokens': 300,
            'temperature': 0.7
        }

        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            out = resp.json()
            try:
                answer = out['choices'][0]['message']['content'].strip()
            except:
                answer = json.dumps(out)[:1000]
            answer = answer.replace('**', '')  # remove Markdown bold
            return jsonify({'answer': answer})
        else:
            return jsonify({'error': 'GenAI API error', 'details': resp.text}), 500

    except Exception as e:
        print("Error in /genai:", str(e))
        return jsonify({'error': str(e)}), 500

# ----------------------------------------
# District / Year / Data lookup
# ----------------------------------------
@app.route('/get_districts', methods=['POST'])
def get_districts():
    try:
        if df is None:
            return jsonify({'error': 'Dataset not available'}), 500
        state = (request.json.get('state') or '').strip()
        if not state:
            return jsonify({'districts': []})
        districts = df[df['STATE/UT'] == state]['DISTRICT'].dropna().unique().tolist()
        return jsonify({'districts': sorted(districts)})
    except Exception as e:
        print("Error in /get_districts:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/get_years', methods=['POST'])
def get_years():
    try:
        if df is None:
            return jsonify({'error': 'Dataset not available'}), 500
        state = (request.json.get('state') or '').strip()
        district = (request.json.get('district') or '').strip()
        if not state or not district:
            return jsonify({'years': []})
        filtered = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district)]
        years = sorted(filtered['Year'].dropna().unique().tolist(), reverse=True)
        return jsonify({'years': years})
    except Exception as e:
        print("Error in /get_years:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/get_district_data', methods=['POST'])
def get_district_data():
    try:
        if df is None:
            return jsonify({'error': 'Dataset not available'}), 500

        body = request.json or {}
        state = (body.get('state') or '').strip()
        district = (body.get('district') or '').strip()
        year = body.get('year', None)

        if not state or not district:
            return jsonify({'error': 'state and district required'}), 400

        filtered = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district)]
        if filtered.empty:
            return jsonify({'error': 'No data found'}), 404

        if year:
            filtered = filtered[filtered['Year'] == int(year)]
            if filtered.empty:
                return jsonify({'error': 'No data found for this year'}), 404
            latest = filtered.iloc[0]
        else:
            latest = filtered.sort_values('Year', ascending=False).iloc[0]

        def safe_get(row, keys, default=0):
            for k in keys:
                if k in row:
                    try:
                        return int(row[k])
                    except:
                        try:
                            return int(float(row[k]))
                        except:
                            return default
            return default

        result = {
            'murder': safe_get(latest, ['Murder']),
            'rape': safe_get(latest, ['Rape']),
            'kidnapping': safe_get(latest, ['Kidnapping and Abduction', 'Kidnapping']),
            'foeticide': safe_get(latest, ['Foeticide']),
            'abetment': safe_get(latest, ['Abetment of suicide', 'Abetment']),
            'procuration': safe_get(latest, ['Procuration of minor girls', 'Procuration']),
            'buying': safe_get(latest, ['Buying of girls for prostitution', 'Buying']),
            'selling': safe_get(latest, ['Selling of girls for prostitution', 'Selling']),
            'child_marriage': safe_get(latest, ['Prohibition of child marriage act', 'Prohibition of child marriage']),
            'other_crimes': safe_get(latest, ['Other Crimes', 'Other']),
            'year': int(latest.get('Year', 0))
        }

        return jsonify(result)

    except Exception as e:
        print("Error in /get_district_data:", str(e))
        return jsonify({'error': str(e)}), 500

# ----------------------------------------
if __name__ == '__main__':
    print("\n🚀 Starting Child Neglect Prediction Platform...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
