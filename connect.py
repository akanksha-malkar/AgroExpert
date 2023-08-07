from flask import Flask, render_template, request
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
  

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

def perform_encoding(df, columns, encoder=None):  
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(df[columns])
    encoded_data = pd.DataFrame(encoder.transform(df[columns]))
    feature_names = encoder.get_feature_names_out(columns)
    encoded_data.columns = feature_names
    return encoded_data, encoder

@app.route('/')
def hello_world():
     return render_template('crop.html')

@app.route('/kolhapur')
def kolhapur():
    return render_template('kolhapur.html')

@app.route('/sangli')
def sangli():
    return render_template('sangli.html')

@app.route('/satare')
def satara():
    return render_template('satara.html')

@app.route('/pune')
def pune():
    return render_template('pune.html')

@app.route('/solapur')
def solapur():
    return render_template('solapur.html')

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    data = pd.read_csv('Crop and fertilizer dataset.csv')

    # Retrieve the form data
    district = request.form['District_Name']
    soil_color = request.form['soil_color']
    nitrogen = request.form['Nitrogen']
    phosphorus = request.form['Phosphorus']
    potassium = request.form['Potassium']
    pH = request.form['pH']
    rainfall = request.form['Rainfall']
    temperature = request.form['Temperature']

    # Create a dataframe from the form data
    input_categorical = pd.DataFrame({
        'District_Name': [district],
        'Soil_color': [soil_color],
    })
    input_numerical = pd.DataFrame({
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'pH': [pH],
        'Rainfall': [rainfall],
        'Temperature': [temperature]
    })

    # Perform the necessary data encoding for the input data (similar to the code you provided)
    X_categorical = data[['District_Name', 'Soil_color']]
    X_numerical = data[['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']]
    categorical_columns = ['District_Name', 'Soil_color']
    X_categorical_encoded, encoder = perform_encoding(X_categorical, categorical_columns)

    # encoder=OneHotEncoder(handle_unknown='ignore', sparse=False, sparse_output=False)
    input_categorical_encoded, _ = perform_encoding(input_categorical, categorical_columns, encoder)

    X_encoded = pd.concat([X_categorical_encoded, X_numerical], axis=1)
    input_encoded = pd.concat([input_categorical_encoded, input_numerical], axis=1)

    # Make predictions using the loaded model
    prediction = model.predict(input_encoded)
    link = data[(data['Crop'] == prediction[0][0]) & (data['Fertilizer'] == prediction[0][1])]['Link'].values[0]
    return render_template('result.html',prediction1=prediction[0][0],prediction2=prediction[0][1],plink=link)

if __name__ == "__main__":
    print("Starting Python Flask Server For crop and fertilizer recommendation system...")
    app.run(debug=True)
