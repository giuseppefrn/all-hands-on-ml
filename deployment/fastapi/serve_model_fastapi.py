
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model (assumes model is saved as 'model.pkl')
model = joblib.load("model.pkl")

# Define request schema
class PredictRequest(BaseModel):
    data: list  # expects a list of lists or a single feature list

@app.post("/predict")
def predict(request: PredictRequest):
    input_array = np.array(request.data)
    if input_array.ndim == 1:
        input_array = input_array.reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}
