from flask import Flask, request, render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

crop_seasons = {
    "Rice": "Summer; Rainy; Autumn; Late Autumn",
    "Maize": "Summer; Rainy; Late Autumn; Spring",
    "Jute": "Rainy; Autumn (harvest)",
    "Cotton": "Summer; Rainy; Late Autumn; Winter",
    "Coconut": "All seasons",
    "Papaya": "All seasons",
    "Orange": "Winter",
    "Apple": "Winter (in cold regions only)",
    "Muskmelon": "Summer",
    "Watermelon": "Summer",
    "Grapes": "Spring",
    "Mango": "Summer (fruiting); Spring (flowering)",
    "Banana": "All seasons",
    "Pomegranate": "Winter; Spring",
    "Lentil": "Autumn (sowing); Late Autumn; Winter",
    "Blackgram": "Rainy; Autumn",
    "Mungbean": "Summer; Autumn; Spring",
    "Mothbeans": "Summer; Rainy",
    "Pigeonpeas": "Rainy; Autumn; Late Autumn",
    "Kidneybeans": "Spring",
    "Chickpea": "Autumn (sowing); Late Autumn; Winter",
    "Coffee": "Rainy (flowering); Late Autumn (harvest)"
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get the form inputs
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    # Log the input values
    print(f"Inputs: N={N}, P={P}, K={K}, Temp={temp}, Humidity={humidity}, pH={ph}, Rainfall={rainfall}")
    
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale the features
    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    
    # Model prediction
    prediction = model.predict(sc_mx_features)
    print(f"Model Prediction: {prediction}")
    
    crop = crop_dict.get(prediction[0], "Unknown")
    season = crop_seasons.get(crop, "Season information not available")

    # Log the predicted crop and its season
    print(f"Predicted Crop: {crop}, Best Season: {season}")
    
    if crop != "Unknown":
        result = f"{crop} is the best crop to be cultivated right there"
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', result=result, crop=crop, season=season)

if __name__ == "__main__":
    app.run(debug=True)
