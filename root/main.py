# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 13:14:53 2024

@author: Admin
"""
import os
print(os.getcwd())

os.chdir(r'D:\Data Science and Data Analytics\ML\Airlines Customer Satisfaction\root')

import streamlit as st
from utils import predict  # Import the prediction function

# Page configuration
st.set_page_config(page_title="Airlines Customer Satisfaction", page_icon="✈️", layout="centered")
st.title("✈️ Airlines Customer Satisfaction Analysis")

# Sidebar Instructions
with st.sidebar:
    st.header("📌 Instructions")
    st.write("""
    1. Fill in the necessary details about the flight experience.
    2. Adjust sliders and dropdowns for precise inputs.
    3. Click 'Calculate Satisfaction' to analyze the results.
    """)
    st.image(r"D:\Data Science and Data Analytics\ML\Airlines Customer Satisfaction\root\ABC_Airlines.JPG", caption="Fly with Comfort")  # Optional logo or image

# Customer Details
st.subheader("👤 Customer Information")
col1, col2 = st.columns(2)

age = col1.number_input("Age", min_value=0, max_value=110, value=35, help="Enter the customer's age.")
travel_class = col2.selectbox("Travel Class", ['Eco', 'Eco Plus', 'Business'], index=0, 
                              help="Select the class of travel (Eco, Eco Plus, Business).")

# Flight Details
st.subheader("✈️ Flight Experience")
col3, col4 = st.columns(2)

flight_distance = col3.number_input("Flight Distance (km)", min_value=0, max_value=20000, value=500, 
                                    help="Enter the distance of the flight in kilometers.")
departure_convenience = col4.slider("Departure/Arrival Time Convenience", min_value=0, max_value=5, value=3, 
                                     help="Rate the convenience of departure/arrival time (0 = Poor, 5 = Excellent).")

col5, col6 = st.columns(2)
gate_location = col5.slider("Gate Location Convenience", min_value=0, max_value=5, value=3, 
                            help="Rate the gate location convenience (0 = Poor, 5 = Excellent).")
wifi_service = col6.slider("Inflight Wi-Fi Service", min_value=0, max_value=5, value=3, 
                            help="Rate the quality of inflight Wi-Fi service (0 = Poor, 5 = Excellent).")

# Service Ratings
st.subheader("🍽️ Onboard Services")
col7, col8, col9 = st.columns(3)

onboard_service = col7.slider("Onboard Service", min_value=0, max_value=5, value=4, 
                               help="Rate the onboard service quality (0 = Poor, 5 = Excellent).")
leg_room_service = col8.slider("Leg Room Comfort", min_value=0, max_value=5, value=4, 
                                help="Rate the comfort of legroom (0 = Poor, 5 = Excellent).")
checkin_service = col9.slider("Check-in Service", min_value=0, max_value=5, value=4, 
                               help="Rate the check-in service quality (0 = Poor, 5 = Excellent).")

# Delay Information
st.subheader("⏱️ Flight Timeliness")
col10, col11 = st.columns(2)

departure_delay = col10.number_input("Departure Delay (Minutes)", min_value=0, max_value=1440, value=0, 
                                     help="Enter the total departure delay in minutes.")
arrival_delay = col11.number_input("Arrival Delay (Minutes)", min_value=0, max_value=1440, value=0, 
                                   help="Enter the total arrival delay in minutes.")

# Customer and Travel Type
st.subheader("🌍 Travel Information")
col12, col13 = st.columns(2)

customer_type = col12.radio("Customer Type", ['Loyal Customer', 'Disloyal Customer'], 
                            help="Select whether the customer is loyal or disloyal.")
travel_type = col13.radio("Type of Travel", ['Business Travel', 'Personal Travel'], 
                          help="Select the purpose of travel.")

# Calculate Satisfaction
if st.button("Calculate Satisfaction"):
    # Call prediction function with individual arguments
    satisfaction_probability, satisfaction_score, satisfaction_level = predict(
        age,
        travel_class,
        flight_distance,
        departure_convenience,
        gate_location,
        wifi_service,
        onboard_service,
        leg_room_service,
        checkin_service,
        departure_delay,
        arrival_delay,
        customer_type,
        travel_type,
    )

    # Display Results
    st.success("✅ Satisfaction Analysis Completed!")
    st.write(f"**Satisfaction Score:** {satisfaction_score:.2f} (Scale: 0-10)")
    st.write(f"**Satisfaction Level:** {satisfaction_level}")

    # Insights based on satisfaction level
    if satisfaction_score < 3:
        st.error("🔴 Poor Satisfaction: Significant improvements are needed.")
    elif 3 < satisfaction_score < 8:
        st.warning("🟠 Moderate Satisfaction: Customer experience is average, improvements suggested.")
    else:
        st.success("🌟 Excellent Satisfaction: Customer experience is exceptional.")
