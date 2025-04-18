from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import joblib

app = Flask(__name__)

model = xgb.XGBRegressor()
model.load_model("commodity_price_model.json")
label_encoders = joblib.load("label_encoders.pkl")

features = ["admin1", "admin2", "market", "category", "commodity",
            "unit", "pricetype", "currency", "year", "month", "day"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    for col in label_encoders:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    prediction = model.predict(df[features])[0]
    return jsonify({'predicted_price': round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

  
