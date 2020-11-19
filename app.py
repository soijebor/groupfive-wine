# Import dependecies
import flask
import numpy as np
import pandas as pd
import sys
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

model = None

# Create a Flask application
app = flask.Flask(__name__)

# Create a function to read scaler and model
def load_model(wine_type):

    # If color is Red read in the scaler and model for red wine
    if wine_type == "Red":

        with open('red_scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)
        with open('red_model.pkl', 'rb') as f: 
            model = pickle.load(f)

    # If color is White read in the scaler and model for  white wine
    elif wine_type == "White":

        with open('white_scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)

        with open('white_model.pkl', 'rb') as f: 
            model = pickle.load(f)

    return model, scaler

# Create a function to read data
def load_data(wine_type):

    # If color is Red load in the red wine encoded dataset
    if wine_type == "Red":

        wine_df = pd.read_csv("Red.csv")

    # If color is White load in the white wine encoded dataset
    elif wine_type == "White":

        wine_df = pd.read_csv("White.csv")

    return wine_df

# Define the home route
@app.route('/',methods=['GET', 'POST'])

# Create a function home() with a return statement
def home():
    if request.method =='GET':
        return render_template('index6.html')

# Create the route for the prediction analysis
@app.route('/predict',methods=['GET', 'POST'])
# Create the prediction() function.
def predict():

    if flask.request.method =='POST':

        color = flask.request.form['color']
        vintage= flask.request.form['vintage']
        vineyard = flask.request.form['vineyard']
        
        wine_df = pd.DataFrame()
        model = []
        final_input = []

        if color == "Red Wine":

            model, scaler = load_model("Red")
            wine_df = load_data(wine_type= "Red")

        elif color == "White Wine":
            
            model, scaler = load_model("White")
            wine_df = load_data(wine_type= "White")

        print(list(wine_df.columns), file=sys.stderr)

        # Create input variable to use for prediction
        wine_input = wine_df.loc[wine_df['vintage'] == int(vintage)]
        prev_input = wine_input

        if len(wine_input) > 0:
            prev_input = wine_input
            wine_input = wine_input.loc[wine_df['vineyard_' + vineyard] == vineyard]
        if len(wine_input) > 0:
            final_input = wine_input # will return wine input
        else:
            final_input = prev_input

        print(final_input, file=sys.stderr)

        # Scale the input
        scaled_final_input = scaler.transform(final_input)

        # Predict scaled input
        prediction = model.predict(scaled_final_input)

        # Find the average of the prediction
        avg_pred = np.mean(prediction)
        if avg_pred > 0.5:
            avg_pred = "High Quality Wine"
        else:
            avg_pred = "Low Quality Wine"

    return jsonify(avg_pred)

# Create the route for the wine type selector analysis
@app.route('/wine_select',methods=['GET'])
# Create the wine_select() function.
def wine_select():

    color = flask.request.form['color']

    if color == "Red Wine":
        wine_df = load_data(wine_type= "Red")

    elif color == "White Wine":
        wine_df = load_data(wine_type= "White")

    wine_columns = [col for col in list(wine_df.columns) if "vineyard" in col]

    return render_template("index6.html", wine_columns)
if __name__ == "__main__":
    app.run(debug=True)
