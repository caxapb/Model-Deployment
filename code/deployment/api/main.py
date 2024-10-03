# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib

# Load the trained model
path = '/app/models/random_forest_model.pkl'

with open(path, "rb") as f:
    model = joblib.load(f)

# Define the FastAPI app
app = FastAPI()

# Define the input data schema
class MushroomInput(BaseModel):
    cap_diameter: float
    cap_shape: float
    gill_attachment: float
    gill_color: float
    stem_height: float
    stem_width: float
    stem_color: float
    season: float


## cap-diameter,cap-shape,gill-attachment,gill-color,stem-height,stem-width,stem-color,season,class
## 1372,2,2,10,3.8074667544799388,1545,11,1.8042727086281731,1

@app.get("/")
async def root():
    return {"message" : "Hello World!!!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: MushroomInput):
    data = [[
        input_data.cap_diameter,
        input_data.cap_shape,
        input_data.gill_attachment,
        input_data.gill_color,
        input_data.stem_height,
        input_data.stem_width,
        input_data.stem_color,
        input_data.season
    ]]
    
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}