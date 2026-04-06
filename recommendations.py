import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_dataframe(df):

    df = df.copy()

    df['brand'] = df['name'].apply(lambda x: x.split()[0])

    df['engine'] = df['engine'].astype(str).str.replace(' CC', '', regex=False)
    df['max_power'] = df['max_power'].astype(str).str.replace(' bhp', '', regex=False)
    df['mileage'] = df['mileage'].astype(str).str.replace(' kmpl', '', regex=False).str.replace(' km/kg', '', regex=False)

    df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

    df["model_family"] = df["name"].apply(lambda x: " ".join(x.split()[:2]))

    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].fillna(
            df.groupby(['name', 'fuel'])[col].transform('mean')
        )
        df[col] = df[col].fillna(
            df.groupby('fuel')[col].transform('median')
        )

    return df

def recommend_car(df, car_name, n=5):

    df = preprocess_dataframe(df)

    features = ['engine', 'mileage', 'seats', 'max_power', 'selling_price']
    X = df[features]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    car_index = df[df['name'] == car_name].index[0]

    similarity_scores = cosine_similarity(
        [X_scaled[car_index]],
        X_scaled
    )[0]

    recommendation = []
    selected = df.iloc[car_index]

    for idx, score in enumerate(similarity_scores):

        if idx == car_index:
            continue

        car = df.iloc[idx]

        if car['model_family'] == selected['model_family']:
            score *= 0.3

        if car['fuel'] == selected['fuel']:
            score += 0.1

        if car['transmission'] == selected['transmission']:
            score += 0.3

        if abs(car['selling_price'] - selected['selling_price']) > 250000:
            score *= 0.8

        recommendation.append((idx, score))

    recommendation = sorted(recommendation, key=lambda x: x[1], reverse=True)

    seen = set()
    final_indices = []

    for idx, score in recommendation:
        name = df.iloc[idx]['model_family']

        if name not in seen:
            seen.add(name)
            final_indices.append(idx)

        if len(final_indices) == n:
            break


    return df.iloc[final_indices][
        ['name', 'fuel', 'transmission', 'selling_price']
    ].drop_duplicates().head(n)