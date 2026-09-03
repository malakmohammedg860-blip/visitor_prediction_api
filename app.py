from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os




app = FastAPI(
    title="Egypt Tourism Smart Recommendation API",
    version="2.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)




BASE_DIR = os.path.dirname(os.path.abspath(__file__))



MODEL_PATH = os.path.join(
    BASE_DIR,
    "visitor_prediction_model.json"
)

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

print("MODEL LOADED SUCCESSFULLY")
print("Model:", MODEL_PATH)

feature_names = model.get_booster().feature_names

print("Number of visitor prediction features:", len(feature_names))




RECOMMENDATION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "tourism_recommendation_model.pkl"
)

recommendation_bundle = joblib.load(
    RECOMMENDATION_MODEL_PATH
)

recommendation_model = recommendation_bundle["model"]

recommendation_feature_names = recommendation_bundle["feature_names"]

print("RECOMMENDATION MODEL LOADED SUCCESSFULLY")
print(
    "Number of recommendation features:",
    len(recommendation_feature_names)
)



DATA_PATH = os.path.join(
    BASE_DIR,
    "egypt_tourism_model_with_place_id (2).xlsx"
)

tourism_df = pd.read_excel(DATA_PATH)

print("TOURISM DATA LOADED SUCCESSFULLY")
print("Dataset shape:", tourism_df.shape)




class PredictionRequest(BaseModel):

    Date: str
    Location: str
    Temperature_C: float
    Is_Holiday: int
    Weather: str
    Time: str
    Event: str




class RecommendationRequest(BaseModel):

    Date: str




@app.post("/predict")
def predict(data: PredictionRequest):

    date = pd.to_datetime(data.Date)

    day_of_week = date.day_name()
    month = date.month

    if month in [12, 1, 2]:

        season = "Winter"

    elif month in [3, 4, 5]:

        season = "Spring"

    elif month in [6, 7, 8]:

        season = "Summer"

    else:

        season = "Autumn"


    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )


    if "Temperature_C" in input_data.columns:

        input_data["Temperature_C"] = data.Temperature_C


    if "Is_Holiday" in input_data.columns:

        input_data["Is_Holiday"] = data.Is_Holiday


    location_column = "Location_" + data.Location

    if location_column in input_data.columns:

        input_data[location_column] = 1


    weather_column = "Weather_" + data.Weather

    if weather_column in input_data.columns:

        input_data[weather_column] = 1


    day_column = "Day_of_week_" + day_of_week

    if day_column in input_data.columns:

        input_data[day_column] = 1


    season_column = "Season_" + season

    if season_column in input_data.columns:

        input_data[season_column] = 1


    time_column = "Time_" + data.Time

    if time_column in input_data.columns:

        input_data[time_column] = 1


    if data.Event != "None":

        event_column = "Event_" + data.Event

        if event_column in input_data.columns:

            input_data[event_column] = 1


    prediction = model.predict(input_data)[0]

    prediction = max(
        0,
        int(round(prediction))
    )


    if prediction < 2283:

        crowd_level = "Low"

    elif prediction <= 4504:

        crowd_level = "Medium"

    else:

        crowd_level = "High"


    if crowd_level == "High":

        recommendation = (
            "The place is expected to be crowded. "
            "Consider visiting another location."
        )

    elif crowd_level == "Medium":

        recommendation = (
            "The place is expected to have moderate crowds. "
            "You can visit, but consider choosing a less busy time."
        )

    else:

        recommendation = (
            "The place is expected to be relatively quiet. "
            "It is a good time to visit."
        )


    return {
        "location": data.Location,
        "date": data.Date,
        "predicted_visitor_count": prediction,
        "crowd_level": crowd_level,
        "recommendation": recommendation
    }



@app.post("/recommend")
def recommend(data: RecommendationRequest):

    selected_date = pd.to_datetime(data.Date)

    tourism_df["Date"] = pd.to_datetime(
        tourism_df["Date"]
    )

    date_data = tourism_df[
        tourism_df["Date"] == selected_date
    ].copy()


    if date_data.empty:

        return {
            "date": data.Date,
            "message": "Date not found in dataset",
            "recommendations": []
        }


 

    recommendation_features = date_data.drop(
        columns=[
            "Recommended",
            "Crowd_Level",
            "Visitor_Count",
            "Date",
            "Source",
            "Data_Type",
            "place_id"
        ],
        errors="ignore"
    )


  

    if "Event" in recommendation_features.columns:

        recommendation_features["Event"] = (
            recommendation_features["Event"]
            .fillna("None")
        )


  

    categorical_columns = [
        "Location",
        "City",
        "Weather",
        "Day_of_week",
        "Season",
        "Time",
        "Event"
    ]


    recommendation_features = pd.get_dummies(
        recommendation_features,
        columns=categorical_columns,
        dtype=int
    )


   

    recommendation_features = (
        recommendation_features.reindex(
            columns=recommendation_feature_names,
            fill_value=0
        )
    )


  

    scores = recommendation_model.predict_proba(
        recommendation_features
    )[:, 1]


    date_data["Recommendation_Score"] = scores


    

    recommendations = (
        date_data[
            [
                "place_id",
                "Location",
                "City",
                "Recommendation_Score"
            ]
        ]
        .sort_values(
            by="Recommendation_Score",
            ascending=False
        )
        .head(5)
    )


    

    result = []

    for _, row in recommendations.iterrows():

        result.append({

            "place_id": int(row["place_id"]),

            "location": row["Location"],

            "city": row["City"],

            "recommendation_score": round(
                float(row["Recommendation_Score"]),
                3
            )

        })


    return {

        "date": data.Date,

        "recommendations": result

    }




@app.get("/")
def root():

    return {

        "message": "Egypt Tourism Smart Recommendation API is running",

        "endpoints": [
            "/predict",
            "/recommend"
        ],

        "docs": "/docs"

    }