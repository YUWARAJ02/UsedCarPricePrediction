import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, session, redirect, url_for, make_response
import pickle as p
import datetime
app=Flask(__name__)

# Set a secret key for the Flask application
app.secret_key = 'key123'  # Replace with a unique and secret key


with open('UCP_model.pkl', 'rb') as file:
    model = p.load(file)

# Load the unique_car_info DataFrame from a CSV file
unique_car_info = pd.read_csv('E:/sql/UsedCarPrice/encodedValues/unique_car_info.csv')

# Load the unique_location_info DataFrame from a CSV file
unique_location_info = pd.read_csv('E:/sql/UsedCarPrice/encodedValues/unique_location_info.csv')

# Load the unique_Fuel_Type_info DataFrame from a CSV file
unique_Fuel_Type_info = pd.read_csv('E:/sql/UsedCarPrice/encodedValues/unique_Fuel_Type_info.csv')

# unique_Owner_Type_info DataFrame from a CSV file
unique_Owner_Type_info = pd.read_csv('E:/sql/UsedCarPrice/encodedValues/unique_Owner_Type_info.csv')

# Assuming unique_car_info is your DataFrame
car_names = unique_car_info.values.tolist()
car_location = unique_location_info.values.tolist()
car_fuel_Type = unique_Fuel_Type_info.values.tolist()
car_Owner_Type = unique_Owner_Type_info.values.tolist()

@app.route('/')
def HOME():
    
    return render_template('index.html', car_names=car_names,
                           car_location=car_location,
                           car_fuel_Type=car_fuel_Type,car_Owner_Type=car_Owner_Type)
    

@app.route('/usedCar_predict',methods=['POST'])
def UsedCar():
    # Your feature list
    feature1 = ['Cars', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission', 
            'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']
    # Collect input from the user
    user_input = []
    form_value=request.form.values() 
    print(form_value)
    for x in form_value:
        user_input.append(pd.to_numeric(x).astype(float))

    # Create a DataFrame with the user input
    user_input_df = pd.DataFrame([user_input], columns=feature1)

    # Make predictions on the user input
    user_prediction = model.predict(user_input_df)

    print(f"Predicted Price: {user_prediction[0]}")
    return render_template('index.html',pred=f'Price {round(user_prediction[0]/10,2)} Lakhs',car_names=car_names,
                           car_location=car_location,
                           car_fuel_Type=car_fuel_Type,car_Owner_Type=car_Owner_Type)



if __name__=='__main__':
    app.run(debug=True)