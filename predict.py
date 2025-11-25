import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model_data = joblib.load('netflix_type_rf_model.pkl')
clf = model_data['pipeline']
le_target = model_data['label_encoder']
feature_cols = model_data['feature_columns']

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Netflix Type Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Make sure we have all required columns
        for col in feature_cols:
            if col not in df.columns:
                df[col] = None
        
        # Select only the features we need
        X = df[feature_cols]
        
        # Make prediction
        prediction = clf.predict(X)[0]
        probability = clf.predict_proba(X)[0]
        
        # Convert prediction to label
        predicted_type = le_target.inverse_transform([prediction])[0]
        confidence = float(max(probability))
        
        # Return result
        return jsonify({
            'prediction': predicted_type,
            'confidence': confidence,
            'probabilities': {
                'Movie': float(probability[0]),
                'TV Show': float(probability[1])
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)