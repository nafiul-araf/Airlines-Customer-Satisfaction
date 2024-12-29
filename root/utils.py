# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 10:37:52 2024

@author: Admin
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import lightgbm


model_data = joblib.load(r"root/model/model_data.pkl")
model_data


# import importlib

# # List of libraries to check
# libraries = [
#     "joblib",
#     "pandas",
#     "numpy",
#     "streamlit",
#     "sklearn",
#     "lightgbm"
# ]

# # Loop through each library and print its version
# for library in libraries:
#     try:
#         module = importlib.import_module(library)
#         print(f"{library} version: {module.__version__}")
#     except ImportError:
#         print(f"{library} is not installed.")
#     except AttributeError:
#         print(f"Unable to determine version for {library}.")





model = model_data['model']
print(model)

scaler = model_data['scaler']
print(scaler)

features = model_data['features']
print(features)
len(features)

columns_to_scale = model_data['cols_to_scale']
print(columns_to_scale)



def data_preparation(age, classes, flight_distance, departure_arrival_time_convenient, gate_location, inflight_wifi_service,
                     onboard_service, leg_room_service, checkin_service, departure_delay_in_minutes,
                     arrival_delay_in_minutes, customer_type, travel_type):
    
    # Define the class mapping
    class_mapping = {'Eco': 1, 'Eco Plus': 2, 'Business': 3}
    
    # Map the classes to numerical values
    classes = class_mapping.get(classes, 1)  # Default to 'Eco' if not found
    
    
    data_input = {
        'Age': age,
        'Class': classes,
        'Flight Distance': flight_distance,
        'Departure/Arrival time convenient': departure_arrival_time_convenient,
        'Gate location': gate_location,
        'Inflight wifi service': inflight_wifi_service,
        'On-board service': onboard_service,
        'Leg room service': leg_room_service,
        'Checkin service': checkin_service,
        'Departure Delay in Minutes': departure_delay_in_minutes,
        'Arrival Delay in Minutes': arrival_delay_in_minutes,
        'Customer Type_disloyal Customer': 1 if customer_type == 'disloyal Customer' else 0,
        'Type of Travel_Personal Travel': 1 if travel_type == 'Personal Travel' else 0
    }

    # Convert the dictionary into a DataFrame
    df = pd.DataFrame([data_input])
    
    # Scale selected columns if needed
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    
    # Ensure the DataFrame includes only the necessary features
    df = df[features]
    
    return df


def calculate_satisfaction_score(input_df):
    satisfaction_probability = model.predict_proba(input_df)[:, 1]  # Probability of being satisfied (class 1)

    # Scale the satisfaction probability to a score between 0 and 10
    satisfaction_score = satisfaction_probability.flatten() * 10

    # Determine the rating category based on the satisfaction score
    def get_rating(score):
        if 0 <= score < 4:
            return 'Poor'
        elif 4 <= score < 8:
            return 'Moderate'
        elif 8 <= score <= 10:
            return 'Excellent'
        else:
            return 'Undefined'  # in case of any unexpected score

    rating = get_rating(satisfaction_score[0])

    return satisfaction_probability.flatten()[0], round(satisfaction_score[0], 2), rating



def predict(age, classes, flight_distance, departure_arrival_time_convenient, gate_location, inflight_wifi_service,
                     onboard_service, leg_room_service, checkin_service, departure_delay_in_minutes,
                     arrival_delay_in_minutes, customer_type, travel_type):

    input_df = data_preparation(age, classes, flight_distance, departure_arrival_time_convenient, gate_location, inflight_wifi_service,
                         onboard_service, leg_room_service, checkin_service, departure_delay_in_minutes,
                         arrival_delay_in_minutes, customer_type, travel_type)

    probability, satisfaction_score, rating = calculate_satisfaction_score(input_df)

    return probability, satisfaction_score, rating


















































