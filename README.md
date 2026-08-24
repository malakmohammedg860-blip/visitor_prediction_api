Visitor Prediction API



A Machine Learning API built with FastAPI to predict visitor counts at Egyptian tourist attractions.



&#x20;Project Overview



This project provides a REST API that uses a trained Machine Learning model to predict the expected number of visitors based on different factors such as:



Date

Tourist Location

Temperature

Weather

Holidays

Day of the Week

Season

Visiting Time



The API is designed to support tourism applications by helping users identify expected visitor levels and make better decisions about when to visit tourist attractions.

&#x20;Technologies

Python

FastAPI

Scikit-learn

XGBoost

Pandas

NumPy

Joblib

Uvicorn

HTML / JavaScript

&#x20;Machine Learning



The project uses a trained Machine Learning model for visitor-count prediction.



The model was developed using:



Data Preprocessing

Feature Engineering

Time-Series Validation

Hyperparameter Tuning

Model Evaluation

Model Files

visitor\_prediction\_model.pkl

visitor\_prediction\_model.json

visitor\_prediction\_metadata.pkl

How to Run

1\. Install dependencies

pip install fastapi uvicorn pandas numpy scikit-learn xgboost joblib

2\. Start the API

uvicorn app:app --reload --host 0.0.0.0 --port 8000

3\. Open the API



Visit:



http://127.0.0.1:8000

&#x20;API Documentation



FastAPI automatically provides interactive API documentation.



Open:



http://127.0.0.1:8000/docs



You can use Swagger UI to test the prediction endpoint directly.



&#x20;Example Input

{

&#x20; "Date": "2026-08-25",

&#x20; "Location": "Pyramids of Giza",

&#x20; "Temperature\_C": 32,

&#x20; "Weather": "Sunny",

&#x20; "Is\_Holiday": false,

&#x20; "Day\_of\_week": "Tuesday",

&#x20; "Season": "Summer"

}

&#x20;Prediction



The API returns a JSON response containing the predicted visitor count.



Example:



{

&#x20; "predicted\_visitors": 1250

}

&#x20;Project Structure

visitor\_prediction\_api/

│

├── app.py

├── index.html

├── visitor\_prediction\_model.pkl

├── visitor\_prediction\_model.json

├── visitor\_prediction\_metadata.pkl

├── README.md

└── .gitignore

&#x20;Goal



The goal of this project is to apply Machine Learning and FastAPI to build a practical tourism prediction system that can help users choose suitable times to visit tourist attractions.



&#x20;Author



Malak Mohamed



AI \& Machine Learning Enthusiast

