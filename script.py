import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fastapi import FastAPI

app = FastAPI()

@app.get('/greet')
def greet():
    return 'Hello, welcome to the recipe site traffic prediction API!'

@app.post('/load_artifact')
async def load_artifact(artifact):
    with open(artifact, 'rb') as file:
        return pkl.load(file)

encoder = load_artifact('encoder.pkl')
scaler = load_artifact('scaler.pkl')
model = load_artifact('forest_model.pkl')

@app.post('/preprocess')
async def preprocess(df):
    df['high_traffic'] = df['high_traffic'].fillna('Low')
    df['servings'] = df['servings'].str.replace(' as a snack', '')
    df['servings'] = df['servings'].astype('int64')
    df['category'] = df['category'].str.replace('Chicken Breast', 'Chicken')
    df['high_traffic_transf'] = encoder.transform(df[['high_traffic']]).astype(int)
    cat_dummy = pd.get_dummies(df['category'], prefix= 'cat', prefix_sep='_',
                           sparse = False, dtype=int)
    combined_df = pd.concat([df, cat_dummy], axis =1)
    features_to_scale = ['calories', 'carbohydrate', 'sugar', 'protein']
    combined_df[features_to_scale] = scaler.transform(combined_df[features_to_scale])
    return combined_df

@app.get('/predict')
def predict(combined_df):
    X = combined_df[['calories', 'carbohydrate', 'sugar', 'protein',
       'servings', 'cat_Beverages', 'cat_Breakfast', 'cat_Chicken', 'cat_Dessert',
       'cat_Lunch/Snacks', 'cat_Meat', 'cat_One Dish Meal', 'cat_Pork', 'cat_Potato', 'cat_Vegetable']]
    y = combined_df['high_traffic_transf']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state=12)
    prediction, prediction_proba = model.predict(X_test), model.predict_proba(X_test)
    return f'prediction: {prediction}, prediction probability: {prediction_proba}'