#flask front end

from flask import Flask, render_template, request
from recommendations import recommend_car
import pandas as pd
import pickle
import locale
import os

locale.setlocale(locale.LC_ALL, 'en_IN')

app = Flask(__name__)

df = pd.read_csv(r"C:\Users\aarya\OneDrive\Desktop\Car_Project\car_seats_filled.csv")

model = pickle.load(open('xgboost.pkl','rb'))
columns = pickle.load(open('columns.pkl','rb'))


df['brand'] = df['name'].str.split().str[0]
brands = sorted(df["brand"].unique())

df['engine'] = df['engine'].str.replace(' CC','').astype(float)
df['max_power'] = df['max_power'].str.replace(' bhp','').astype(float)
df['mileage'] = df['mileage'].str.replace(' kmpl','').str.replace(' km/kg','').astype(float)


@app.route("/", methods=['GET', 'POST'])
def home():

    price = None
    recommendations = None
    engine = None
    mileage = None

    models = []
    brand = None
    model_name = None
    fuel = None
    transmission = None
    year = None
    km = None
    max_power=None
    image_file = None


    if request.method == "POST":

        action = request.form.get('action')
        brand = request.form.get('brand')

        if action == "search":

            if brand:
                models = sorted(df[df["brand"] == brand]["name"].unique())

        if action=='predict':
            model_name = request.form['model']
            models = sorted(df[df["brand"] == brand]["name"].unique())

            image_name = "_".join(model_name.split()[:2]).lower()

            IMAGE_DIR = "static/car_images"


            image_path_png = os.path.join(IMAGE_DIR, f"{image_name}.png")
            image_path_jpg = os.path.join(IMAGE_DIR, f"{image_name}.jpg")

            if os.path.exists(image_path_png):
                image_file = f"car_images/{image_name}.png"

            elif os.path.exists(image_path_jpg):
                image_file = f"car_images/{image_name}.jpg"

            
            year = int(request.form['year'])
            km = int(request.form['kilometer'])
            fuel = request.form['fuel']
            transmission = request.form['transmission']

            car = df[df['name'] == model_name].iloc[0]

            mileage = car["mileage"]
            engine = car["engine"]
            max_power = car["max_power"]
            seats = car["seats"]
            seller_type = car["seller_type"]
            owner = car["owner"]

            fuel_map = {
                'Diesel':1, 
                'Petrol':2,
                'LPG':3, 
                'CNG':4}

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

            brand_map = {b:i for i,b in enumerate(df['brand'].unique())}


            input_data = pd.DataFrame({
                "year": [year],
                "km_driven": [km],
                "fuel": [fuel_map[fuel]],
                "seller_type": [seller_map[seller_type]],
                "transmission": [trans_map[transmission]],
                "owner": [owner_map[owner]],
                "mileage": [mileage],
                "engine": [engine],
                "max_power": [max_power],
                "seats": [seats],
                "brand": [brand_map[brand]]
            })

            input_data = input_data[columns]

            price = int(model.predict(input_data)[0])
            price = locale.format_string("%d", price, grouping=True)

            recommendations = recommend_car(df,model_name)

            for idx, row in recommendations.iterrows():

                image_name = "_".join(row["name"].split()[:2]).lower()

                image_png = f"static/car_images/{image_name}.png"
                image_jpg = f"static/car_images/{image_name}.jpg"

                if os.path.exists(image_png):
                    recommendations.loc[idx, "image"] = f"car_images/{image_name}.png"

                elif os.path.exists(image_jpg):
                    recommendations.loc[idx, "image"] = f"car_images/{image_name}.jpg"

                else:
                    recommendations.loc[idx, "image"] = None
                    recommendations.loc[idx, "expected_image"] = f"{image_name}.png / {image_name}.jpg"

    return render_template(
        "home.html",
        brands=brands,
        models=models,
        fuels = df["fuel"].unique(),
        transmissions = df["transmission"].unique(),
        engine = engine,
        mileage = mileage,
        max_power = max_power,
        price=price,
        recommendations=recommendations,

        selected_brand=brand,
        selected_model=model_name,
        selected_fuel=fuel,
        selected_transmission=transmission,
        selected_year=year,
        selected_km=km,

        image_file=image_file
    )


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')