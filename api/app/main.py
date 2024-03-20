import PIL
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from utils.model_skin import class_id_to_label, load_model, transform_image
from utils.model_text import load_model_text

model = None 
app = FastAPI()


# Create class of answer: only class name 
class ImageClass(BaseModel):
    prediction: str

class SentimentResponse(BaseModel):
    text: str
    sentiment_label: str
    sentiment_score: float

# Load model at startup
@app.on_event("startup")
def startup_event():
    global skin_model, text_model
    skin_model = load_model()
    text_model = load_model_text()

@app.get('/')
def return_info():
    return 'Hello FastAPI'

@app.post('/classify')
def classify(file: UploadFile = File(...)):
    image = PIL.Image.open(file.file)
    adapted_image = transform_image(image)
    pred_index = torch.sigmoid(skin_model(adapted_image.unsqueeze(0))).round().item()
    pred_class = class_id_to_label(pred_index)
    response = ImageClass(
        prediction=pred_class
    )
    return response

# @app.post('/clf_text')
# def predict_sentiment(text: str):
#     text = text.get('text')  # Extract text from the request data
#     if text is None:
#         return {"error": "Text field is missing in request data"}

#     # Here you would perform your sentiment analysis with the received text
#     # Replace this with your actual sentiment analysis code
#     sentiment = text_model(text)

#     response = SentimentResponse(
#         text=text,
#         sentiment_label=sentiment.label,
#         sentiment_score=sentiment.score,
#     )
#     return response


##### run from api folder:
##### uvicorn app.main:app