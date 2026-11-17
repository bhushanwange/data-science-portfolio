from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load your existing saved model
model_filename = 'house_price_model.pkl'
if os.path.exists(model_filename):
    model = joblib.load(model_filename)
    print(f"Loaded existing model from {model_filename}")
else:
    raise FileNotFoundError(f"Model file '{model_filename}' not found! Run train_model.py first.")

@app.route('/')
def home():
    return jsonify({
        "message": "California House Price Prediction API is live!",
        "usage": "Send a POST request to /predict with JSON containing the 8 California housing features."
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
            
        # Parse incoming JSON into a pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Required features for the California Housing dataset
        required_features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup', 'Latitude', 'Longitude'
        ]
        
        for feat in required_features:
            if feat not in input_df.columns:
                return jsonify({"error": f"Missing required feature: {feat}"}), 400
                
        input_df = input_df[required_features]
        
        # Run prediction using your pre-trained model
        prediction = model.predict(input_df)
        
        return jsonify({
            "success": True,
            "predicted_med_house_val": round(float(prediction[0]), 4)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)