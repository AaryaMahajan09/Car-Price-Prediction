#streamlit front end

import streamlit as st
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import os
import re
from recommendations import recommend_car

df = pd.read_csv(r'C:\Users\aarya\OneDrive\Desktop\Car_Project\car_seats_filled.csv')

df['brand'] = df['name'].str.split().str[0]
df['mileage'] = df['mileage'].str.split().str[0].astype(float)
df['engine'] = df['engine'].str.split().str[0].astype(float)
df['max_power'] = df['max_power'].str.split().str[0].astype(float)


model = pickle.load(open('xgboost.pkl','rb'))
columns = pickle.load(open('columns.pkl','rb'))

st.title('Car Price Prediction App')

brand = st.selectbox('Brand Name', sorted(df['brand'].unique()))

brand_car = df[df['brand'] == brand]['name'].unique()
name = st.selectbox('Car Name', sorted(brand_car))


image_name = "_".join(name.split()[:2]).lower()

IMAGE_DIR = 'car_images'

image_path_png = os.path.join(IMAGE_DIR, f"{image_name}.png")
image_path_jpg = os.path.join(IMAGE_DIR, f"{image_name}.jpg")

if os.path.exists(image_path_png):
    # --- FIX 1 ---
    st.image(image_path_png, use_container_width=True)
elif os.path.exists(image_path_jpg):
    # --- FIX 2 ---
    st.image(image_path_jpg, use_container_width=True)
else:
    st.info(f"📷 Image not found. Add '{image_name}.jpg' or .png to the '{IMAGE_DIR}' folder.")




year = st.slider("Year", 2000, 2025, 2016)
km_driven = st.slider("KM Driven", 0, int(df['km_driven'].max()))

fuel = st.selectbox('Fuel Type', sorted(df['fuel'].unique()))
seller_type = st.selectbox('Seller Type', sorted(df['seller_type'].unique()))
transmission = st.selectbox('Transmission', sorted(df['transmission'].unique()))
mileage = st.slider("Mileage", int(df['mileage'].min()), int(df['mileage'].max()))
engine = st.slider("Engine CC", int(df['engine'].min()), int(df['engine'].max()), 900)
max_power = st.slider("Max Power", int(df['max_power'].min()), int(df['max_power'].max()))

owner = st.selectbox('Owner', sorted(df['owner'].unique()))
seats = st.selectbox('Seats', sorted(df['seats'].unique()))


fuel_map = {'Diesel':1, 'Petrol':2, 'LPG':3, 'CNG':4}

seller_map = {
    'Individual':1,
    'Dealer':2,
    'Trustmark Dealer':3
}

trans_map = {
    'Manual':1,
    'Automatic':2
}

owner_map = {
    'First Owner':1,
    'Second Owner':2,
    'Third Owner':3,
    'Fourth & Above Owner':4,
    'Test Drive Car':5
}


input_dict = { 
    'year': year,
    'km_driven': km_driven,
    'fuel': fuel_map[fuel],
    'seller_type': seller_map[seller_type],
    'transmission': trans_map[transmission],
    'owner': owner_map[owner],
    'mileage': mileage,
    'engine': engine,
    'max_power': max_power,
    'seats': seats,
    }



input_df = pd.DataFrame([input_dict])

input_df = input_df.reindex(columns=columns, fill_value=0)


if st.button("Predict Price"):

    price = model.predict(input_df)[0]
    st.success(f"Predicted Price: ₹ {price:.0f}")